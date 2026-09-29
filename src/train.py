from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import as_posix, ensure_output_dirs, load_config
from .features import model_feature_columns
from .io import connect, get_logger, parquet_scan
from .leakage import run_leakage_audit
from .modeling import feature_groups, fit_model, make_model, predict_probability


META_COLUMNS = [
    "timestamp",
    "cuba",
    "is_valid_prediction_row",
    "next_target_source_60m",
    "RMFC",
    "UMFC",
    "SMRWFC",
    "TALIMFC",
    "rmfc_diff_30m",
]


def _numeric_features(path: Path) -> tuple[list[str], dict[str, str]]:
    schema = pq.ParquetFile(path).schema_arrow
    candidates = model_feature_columns(schema.names)
    numeric = []
    types = {}
    for name in candidates:
        arrow_type = schema.field(name).type
        if pa.types.is_integer(arrow_type) or pa.types.is_floating(arrow_type) or pa.types.is_boolean(arrow_type):
            numeric.append(name)
            types[name] = str(arrow_type)
    return numeric, types


def _load_training_sample(
    config: dict[str, Any], features: list[str], horizon: int
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    processed = Path(config["project"]["processed_dir"])
    train_files = as_posix(processed / "features_2026-0[1-5].parquet")
    scan = f"read_parquet('{train_files}', union_by_name=true)"
    fraction = float(config["training"]["negative_sample_fraction"])
    seed = int(config["project"]["seed"])
    target = f"target_ea_{horizon}m"
    feature_sql = ", ".join(f'"{name}"' for name in features)
    threshold = int(round(fraction * 1_000_000))
    con = connect(config)
    query = f"""
        SELECT {feature_sql}, {target}
        FROM {scan}
        WHERE is_valid_prediction_row = 1
          AND (
            {target} = 1
            OR hash(cuba, timestamp, {seed + horizon}) % 1000000 < {threshold}
          )
    """
    frame = con.execute(query).fetchdf()
    con.close()
    y = frame.pop(target).to_numpy(dtype=np.int8)
    x = frame.replace([np.inf, -np.inf], np.nan)
    weights = np.where(y == 1, 1.0, 1.0 / fraction).astype(np.float64)
    summary = {
        "horizon": horizon,
        "sample_rows": int(len(y)),
        "sample_positives": int(y.sum()),
        "sample_prevalence": float(y.mean()),
        "negative_sample_fraction": fraction,
        "weighted_prevalence": float(np.average(y, weights=weights)),
    }
    return x, y, weights, summary


def _score_file(path: Path, model, features: list[str], horizon: int) -> tuple[np.ndarray, np.ndarray]:
    target = f"target_ea_{horizon}m"
    ys: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    columns = features + [target, "is_valid_prediction_row"]
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas()
        valid = frame.pop("is_valid_prediction_row").to_numpy(dtype=bool)
        y = frame.pop(target).to_numpy(dtype=np.int8)[valid]
        if not valid.any():
            continue
        x = frame.loc[valid, features].replace([np.inf, -np.inf], np.nan)
        ys.append(y)
        probabilities.append(predict_probability(model, x))
    return np.concatenate(ys), np.concatenate(probabilities)


def _metrics(y: np.ndarray, probability: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    result = {
        "n": int(y.size),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }
    if threshold is not None:
        prediction = probability >= threshold
        result.update(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y, prediction, zero_division=0)),
                "recall": float(recall_score(y, prediction, zero_division=0)),
                "f1": float(f1_score(y, prediction, zero_division=0)),
            }
        )
    return result


def _best_f1_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if thresholds.size == 0:
        return 1.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[int(np.nanargmax(f1))])


def _train_one(
    algorithm: str,
    config: dict[str, Any],
    x: pd.DataFrame,
    y: np.ndarray,
    weights: np.ndarray,
    features: list[str],
    seed_offset: int = 0,
):
    model = make_model(algorithm, config, int(config["project"]["seed"]) + seed_offset)
    return fit_model(model, algorithm, x[features], y, weights)


def _write_validation_predictions(
    path: Path,
    output: Path,
    models: dict[int, Any],
    features: list[str],
) -> None:
    target_columns = [f"target_ea_{h}m" for h in sorted(models)]
    columns = list(dict.fromkeys(META_COLUMNS + target_columns + features))
    parquet = pq.ParquetFile(path)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
            frame = batch.to_pandas()
            x = frame[features].replace([np.inf, -np.inf], np.nan)
            result = frame[META_COLUMNS + target_columns].copy()
            for horizon, model in sorted(models.items()):
                result[f"risk_{horizon}m"] = predict_probability(model, x).astype(np.float32)
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _plot_validation_curves(predictions_path: Path, reports_dir: Path) -> None:
    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    palette = ["#164E63", "#287C8E", "#65A7B5"]
    line_styles = ["-", "--", ":"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    for color, style, horizon in zip(palette, line_styles, (15, 30, 60)):
        table = pq.read_table(
            predictions_path,
            columns=[f"target_ea_{horizon}m", f"risk_{horizon}m", "is_valid_prediction_row"],
        )
        valid = table["is_valid_prediction_row"].to_numpy().astype(bool)
        y = table[f"target_ea_{horizon}m"].to_numpy()[valid]
        p = table[f"risk_{horizon}m"].to_numpy()[valid]
        precision, recall, _ = precision_recall_curve(y, p)
        step = max(1, len(precision) // 2500)
        ax.plot(recall[::step], precision[::step], color=color, linestyle=style, linewidth=1.8,
                label=f"{horizon} min · AP={average_precision_score(y, p):.3f}")
        ax.axhline(y.mean(), color=color, linestyle=(0, (1, 4)), linewidth=0.7, alpha=0.5)
    ax.set_title("Curvas precision–recall por horizonte", loc="left", pad=24)
    ax.text(0, 1.01, "Validación: junio 2026 · distribución real · filas válidas", transform=ax.transAxes,
            fontsize=9, color="#4B5563", va="bottom")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "precision_recall_validacion.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    curves = []
    for color, style, horizon in zip(palette, line_styles, (15, 30, 60)):
        table = pq.read_table(
            predictions_path,
            columns=[f"target_ea_{horizon}m", f"risk_{horizon}m", "is_valid_prediction_row"],
        )
        valid = table["is_valid_prediction_row"].to_numpy().astype(bool)
        y = table[f"target_ea_{horizon}m"].to_numpy()[valid]
        p = table[f"risk_{horizon}m"].to_numpy()[valid]
        observed, predicted = calibration_curve(y, p, n_bins=12, strategy="quantile")
        curves.append((color, style, horizon, predicted, observed))
    axis_max = max(
        0.05,
        1.08 * max(max(float(np.max(predicted)), float(np.max(observed))) for _, _, _, predicted, observed in curves),
    )
    ax.plot([0, axis_max], [0, axis_max], color="#374151", linestyle="--", linewidth=1.2, label="Ideal")
    for color, style, horizon, predicted, observed in curves:
        ax.plot(predicted, observed, color=color, linestyle=style, marker="o", markersize=3,
                linewidth=1.6, label=f"{horizon} min")
    ax.set_title("Calibración por horizonte", loc="left", pad=24)
    ax.text(0, 1.01, "Validación: junio 2026 · 12 bins por cuantiles · filas válidas", transform=ax.transAxes,
            fontsize=9, color="#4B5563", va="bottom")
    ax.set_xlabel("Riesgo promedio predicho")
    ax.set_ylabel("Frecuencia observada")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.grid(True, color="#E5E7EB", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "calibracion_validacion.png", bbox_inches="tight")
    plt.close(fig)


def train(config: dict[str, Any]) -> dict[str, Any]:
    if (Path(config['project']['processed_dir'])/'ST6_INVALIDATED.json').exists():
        raise RuntimeError('ST6: reconstruction was stopped; do not train on the invalidated old features. See experiments/st6_temporal_fix/impacto_st6.md.')
    ensure_output_dirs(config)
    run_leakage_audit(config)
    logger = get_logger("train", config)
    processed = Path(config["project"]["processed_dir"])
    models_dir = Path(config["project"]["models_dir"])
    reports_dir = Path(config["project"]["reports_dir"])
    validation_path = processed / "features_2026-06.parquet"
    if not validation_path.exists():
        raise FileNotFoundError("Falta features_2026-06.parquet; ejecutar src.build_dataset.")
    if (models_dir / "test_evaluated.lock").exists():
        raise RuntimeError("El test de julio ya fue abierto. No se permite reentrenar sin revisión explícita.")

    all_features, feature_types = _numeric_features(validation_path)
    groups = feature_groups(all_features)
    comparison_horizon = int(config["training"]["comparison_horizon_minutes"])
    x60, y60, w60, sample60 = _load_training_sample(config, all_features, comparison_horizon)
    logger.info("Muestra train 60m: %s", sample60)

    comparison_rows: list[dict[str, Any]] = []
    comparison_models: dict[str, Any] = {}
    for index, algorithm in enumerate(config["training"]["algorithms"]):
        started = time.time()
        logger.info("Entrenando comparación: %s", algorithm)
        model = _train_one(algorithm, config, x60, y60, w60, all_features, seed_offset=index)
        y_val, p_val = _score_file(validation_path, model, all_features, comparison_horizon)
        row = {"algorithm": algorithm, **_metrics(y_val, p_val), "fit_score_seconds": time.time() - started}
        comparison_rows.append(row)
        comparison_models[algorithm] = model
        joblib.dump(model, models_dir / f"comparison_60m_{algorithm}.joblib", compress=3)
        logger.info("%s PR-AUC=%.5f", algorithm, row["pr_auc"])

    comparison = pd.DataFrame(comparison_rows).sort_values("pr_auc", ascending=False)
    comparison.to_csv(reports_dir / "comparacion_modelos.csv", index=False, encoding="utf-8-sig")
    selected_algorithm = str(comparison.iloc[0].algorithm)
    logger.info("Algoritmo seleccionado en junio: %s", selected_algorithm)

    ablation_rows: list[dict[str, Any]] = []
    ablation_models: dict[str, Any] = {}
    for index, (group_name, group_features) in enumerate(groups.items()):
        started = time.time()
        if group_name == "todas_incluye_vitm":
            model = comparison_models[selected_algorithm]
        else:
            model = _train_one(
                selected_algorithm, config, x60, y60, w60, group_features, seed_offset=100 + index
            )
        y_val, p_val = _score_file(validation_path, model, group_features, comparison_horizon)
        row = {
            "feature_set": group_name,
            "n_features": len(group_features),
            **_metrics(y_val, p_val),
            "fit_score_seconds": time.time() - started,
        }
        ablation_rows.append(row)
        ablation_models[group_name] = model
        logger.info("Ablación %s PR-AUC=%.5f", group_name, row["pr_auc"])
    ablations = pd.DataFrame(ablation_rows).sort_values("pr_auc", ascending=False)
    ablations.to_csv(reports_dir / "ablaciones_features.csv", index=False, encoding="utf-8-sig")
    selected_group = str(ablations.iloc[0].feature_set)
    selected_features = groups[selected_group]
    logger.info("Set de features seleccionado: %s (%d)", selected_group, len(selected_features))

    horizons = [int(value) for value in config["features"]["target_horizons_minutes"]]
    final_models: dict[int, Any] = {}
    sample_summaries: dict[int, dict[str, Any]] = {comparison_horizon: sample60}
    for horizon in horizons:
        if horizon == comparison_horizon:
            model = ablation_models[selected_group]
        else:
            x, y, weights, sample_summary = _load_training_sample(config, selected_features, horizon)
            sample_summaries[horizon] = sample_summary
            model = _train_one(
                selected_algorithm, config, x, y, weights, selected_features, seed_offset=1000 + horizon
            )
        final_models[horizon] = model
        joblib.dump(model, models_dir / f"model_{horizon}m.joblib", compress=3)

    predictions_path = processed / "predictions_validation_2026-06.parquet"
    _write_validation_predictions(validation_path, predictions_path, final_models, selected_features)

    thresholds: dict[str, float] = {}
    validation_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        table = pq.read_table(
            predictions_path,
            columns=[f"target_ea_{horizon}m", f"risk_{horizon}m", "is_valid_prediction_row"],
        )
        valid = table["is_valid_prediction_row"].to_numpy().astype(bool)
        y = table[f"target_ea_{horizon}m"].to_numpy()[valid].astype(np.int8)
        p = table[f"risk_{horizon}m"].to_numpy()[valid].astype(np.float64)
        threshold = _best_f1_threshold(y, p)
        thresholds[str(horizon)] = threshold
        validation_rows.append({"horizon_minutes": horizon, **_metrics(y, p, threshold)})
    validation_metrics = pd.DataFrame(validation_rows)
    validation_metrics.to_csv(
        reports_dir / "metricas_validacion_filas.csv", index=False, encoding="utf-8-sig"
    )
    _plot_validation_curves(predictions_path, reports_dir)

    schema_payload = {
        "feature_order": selected_features,
        "feature_types": {name: feature_types[name] for name in selected_features},
        "excluded_identifiers_and_targets": sorted(set(pq.ParquetFile(validation_path).schema_arrow.names) - set(selected_features)),
    }
    (models_dir / "feature_schema.json").write_text(
        json.dumps(schema_payload, indent=2), encoding="utf-8"
    )
    metadata = {
        "calendar_contract": "ST6=FT_TURNO+30h; reconstructed from source",
        "dataset_manifest": json.loads((processed / "dataset_manifest.json").read_text(encoding="utf-8")),
        "selected_algorithm": selected_algorithm,
        "selected_feature_set": selected_group,
        "n_features": len(selected_features),
        "thresholds_selected_on_june": thresholds,
        "train_period": "2026-01-01/2026-05-31",
        "validation_period": "2026-06-01/2026-06-30",
        "test_period_locked": "2026-07-01/2026-07-31",
        "sample_summaries": sample_summaries,
        "algorithm_comparison": comparison.to_dict(orient="records"),
        "feature_ablations": ablations.to_dict(orient="records"),
        "validation_metrics": validation_metrics.to_dict(orient="records"),
        "target_caveat": "THISEVT oficial sólo 2026-04-01 a 2026-06-02; fuera se usa proxy validado.",
    }
    metadata_path = models_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")

    report = f"""# Entrenamiento y selección — validación junio 2026

## Selección

- Algoritmo: **{selected_algorithm}**.
- Set de variables: **{selected_group}** ({len(selected_features)} features).
- Train: enero–mayo; validación y umbrales: junio; julio no fue leído por este comando.

## Comparación de algoritmos (horizonte {comparison_horizon} min)

{comparison.to_markdown(index=False)}

## Ablaciones de variables (horizonte {comparison_horizon} min)

{ablations.to_markdown(index=False)}

## Métricas por fila y umbrales elegidos en junio

{validation_metrics.to_markdown(index=False)}

## Interpretación provisional

PR-AUC es la métrica principal por el desbalance extremo. Los umbrales maximizan F1 por fila como punto de partida; la evaluación operativa posterior aplica persistencia, cooldown, falsas alarmas por cuba-día y recall top-k. La calibración se reporta, pero no se fuerza una conclusión positiva.

## Limitación obligatoria

Junio combina 92 episodios oficiales y 1.021 proxy. Por eso la selección es válida para continuar un MVP exploratorio, pero no equivale a validación íntegramente oficial.
"""
    (reports_dir / "entrenamiento_validacion.md").write_text(report, encoding="utf-8")
    logger.info("Entrenamiento completo. Metadata: %s", metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y selecciona modelos sin abrir julio.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = train(load_config(args.config))
    print(json.dumps({
        "selected_algorithm": result["selected_algorithm"],
        "selected_feature_set": result["selected_feature_set"],
        "thresholds": result["thresholds_selected_on_june"],
    }, indent=2))


if __name__ == "__main__":
    main()
