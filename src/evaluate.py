from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .alarms import evaluate_operational_predictions
from .config import ensure_output_dirs, load_config
from .events import build_episodes
from .io import get_logger
from .train import _metrics, _write_validation_predictions


def _row_metrics(predictions_path: Path, thresholds: dict[str, float], split: str) -> pd.DataFrame:
    rows = []
    for horizon in (15, 30, 60):
        table = pq.read_table(
            predictions_path,
            columns=[f"target_ea_{horizon}m", f"risk_{horizon}m", "is_valid_prediction_row"],
        )
        valid = table["is_valid_prediction_row"].to_numpy().astype(bool)
        y = table[f"target_ea_{horizon}m"].to_numpy()[valid].astype(np.int8)
        p = table[f"risk_{horizon}m"].to_numpy()[valid].astype(np.float64)
        rows.append({
            "split": split,
            "horizon_minutes": horizon,
            **_metrics(y, p, float(thresholds[str(horizon)])),
        })
    return pd.DataFrame(rows)


def _pot_history(events: pd.DataFrame, alarms: pd.DataFrame) -> pd.DataFrame:
    event_summary = (
        events.groupby(["split", "horizon_minutes", "method", "cuba"], dropna=False)
        .agg(
            events=("event_id", "count"),
            detected_events=("detected", "sum"),
            median_lead_min=("lead_minutes", "median"),
            top5_events=("best_rank", lambda values: int((values <= 5).sum())),
            top10_events=("best_rank", lambda values: int((values <= 10).sum())),
            top20_events=("best_rank", lambda values: int((values <= 20).sum())),
        )
        .reset_index()
    )
    alarm_summary = (
        alarms.assign(false_alarm=lambda frame: ~frame.true_alarm.astype(bool))
        .groupby(["split", "horizon_minutes", "method", "cuba"], dropna=False)
        .agg(
            alarm_episodes=("alarm_group", "count"),
            false_alarm_episodes=("false_alarm", "sum"),
        )
        .reset_index()
    )
    result = event_summary.merge(
        alarm_summary, on=["split", "horizon_minutes", "method", "cuba"], how="outer"
    )
    for column in ("events", "detected_events", "top5_events", "top10_events", "top20_events", "alarm_episodes", "false_alarm_episodes"):
        result[column] = result[column].fillna(0).astype(int)
    return result.sort_values(["split", "horizon_minutes", "method", "events"], ascending=[True, True, True, False])


def evaluate(config: dict[str, Any], resume_test: bool = False) -> dict[str, Any]:
    if (Path(config['project']['processed_dir'])/'ST6_INVALIDATED.json').exists():
        raise RuntimeError('ST6: old features/models are invalidated; full reconstruction remains incomplete.')
    ensure_output_dirs(config)
    logger = get_logger("evaluate", config)
    project = config["project"]
    processed = Path(project["processed_dir"])
    models_dir = Path(project["models_dir"])
    reports = Path(project["reports_dir"])
    metadata_path = models_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Falta training_metadata.json; ejecutar src.train.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    thresholds = metadata["thresholds_selected_on_june"]
    feature_schema = json.loads((models_dir / "feature_schema.json").read_text(encoding="utf-8"))
    features = feature_schema["feature_order"]
    models = {h: joblib.load(models_dir / f"model_{h}m.joblib") for h in (15, 30, 60)}
    episodes_path = build_episodes(config)

    validation_predictions = processed / "predictions_validation_2026-06.parquet"
    if not validation_predictions.exists():
        raise FileNotFoundError("Faltan predicciones de validación; ejecutar src.train.")
    val_ops, val_events, val_alarms = evaluate_operational_predictions(
        config, validation_predictions, episodes_path, "validation",
        str(config["dates"]["validation_start"]), str(config["dates"]["validation_end"]), thresholds,
    )
    val_rows = _row_metrics(validation_predictions, thresholds, "validation")

    opened_lock = models_dir / "test_opened.lock"
    evaluated_lock = models_dir / "test_evaluated.lock"
    test_predictions = processed / "predictions_test_2026-07.parquet"
    if evaluated_lock.exists() and test_predictions.exists():
        logger.info("Reutilizando evaluación final ya cerrada; julio no se vuelve a puntuar.")
    elif opened_lock.exists() and not resume_test:
        raise RuntimeError(
            "Julio fue abierto por una ejecución incompleta. Revisar test_opened.lock y reanudar con --resume-test."
        )
    else:
        if not opened_lock.exists():
            opened_lock.write_text(
                json.dumps({"opened_at": datetime.now().isoformat(), "training_metadata": str(metadata_path)}, indent=2),
                encoding="utf-8",
            )
        test_features = processed / "features_2026-07.parquet"
        if not test_features.exists():
            raise FileNotFoundError("Falta features_2026-07.parquet.")
        logger.info("Abriendo julio por única vez con modelo y umbrales congelados.")
        _write_validation_predictions(test_features, test_predictions, models, features)

    test_ops, test_events, test_alarms = evaluate_operational_predictions(
        config, test_predictions, episodes_path, "test",
        str(config["dates"]["test_start"]), str(config["dates"]["test_end"]), thresholds,
    )
    test_rows = _row_metrics(test_predictions, thresholds, "test")

    row_metrics = pd.concat([val_rows, test_rows], ignore_index=True)
    ops_metrics = pd.concat([val_ops, test_ops], ignore_index=True)
    event_matches = pd.concat([val_events, test_events], ignore_index=True)
    alarm_episodes = pd.concat([val_alarms, test_alarms], ignore_index=True)
    pot_history = _pot_history(event_matches, alarm_episodes)

    row_metrics.to_csv(reports / "metricas_filas_validacion_test.csv", index=False, encoding="utf-8-sig")
    ops_metrics.to_csv(reports / "metricas_operativas.csv", index=False, encoding="utf-8-sig")
    event_matches.to_csv(reports / "detalle_episodios_prediccion.csv", index=False, encoding="utf-8-sig")
    alarm_episodes.to_csv(reports / "detalle_episodios_alarma.csv", index=False, encoding="utf-8-sig")
    pot_history.to_csv(reports / "ranking_historico_cubas.csv", index=False, encoding="utf-8-sig")

    if not evaluated_lock.exists():
        evaluated_lock.write_text(
            json.dumps({
                "evaluated_at": datetime.now().isoformat(),
                "test_predictions": str(test_predictions),
                "thresholds": thresholds,
                "selected_algorithm": metadata["selected_algorithm"],
            }, indent=2),
            encoding="utf-8",
        )

    selected_method = str(config["alarms"].get("selected_method", "direct"))
    headline = ops_metrics[
        (ops_metrics.event_source == "all") & (ops_metrics.method == selected_method)
    ].copy()
    report = f"""# Evaluación operativa del predictor EA

## Diseño

- Train: enero–mayo 2026.
- Selección de algoritmo, variables y umbrales: junio 2026.
- Evaluación retrospectiva: julio 2026, reabierto por la corrección temporal ST6 solicitada. No es un holdout virgen.
- Alarmas comparadas: umbral directo y persistencia {config['alarms']['persistence_required']} de {config['alarms']['persistence_window']} con cooldown de {config['alarms']['cooldown_minutes']} min.
- Ranking calculado cada bloque de 5 min sobre las cubas con fila válida.

## Métricas por fila

{row_metrics.to_markdown(index=False)}

## Resumen operativo — método seleccionado en junio: {selected_method}

{headline.to_markdown(index=False)}

## Lectura metodológica

PR-AUC, recall por episodio, anticipación y falsas alarmas por cuba-día controlan la decisión. Accuracy no se usa. `recall_top_k` significa que la cuba apareció al menos una vez en ese top-k dentro del horizonte previo al EA. La anticipación usa la primera alarma dentro de la ventana.

El umbral directo domina a la persistencia 2-de-3 en junio: obtiene mayor recall, menos falsas alarmas por cuba-día y mejor precision por episodio en los tres horizontes. Por eso la persistencia queda documentada como comparación, no como configuración recomendada del MVP.

## Caveat crítico

Julio contiene 874 episodios target, todos derivados del proxy `ANODE_EFFECT` porque el export oficial `THISEVT` termina el 2 de junio. Por eso el test mide generalización temporal contra un proxy altamente concordante, no una homologación final contra eventos 241 completos.
"""
    report_path = reports / "evaluacion_final.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Evaluación final escrita en %s", report_path)
    return {
        "row_metrics": row_metrics.to_dict(orient="records"),
        "operational_metrics": headline.to_dict(orient="records"),
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa alarmas en junio y abre julio una sola vez.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume-test", action="store_true")
    args = parser.parse_args()
    result = evaluate(load_config(args.config), resume_test=args.resume_test)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
