from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import as_posix, ensure_output_dirs, load_config, source_path
from .io import connect, parquet_scan
from .modeling import fit_model, make_model, predict_probability
from .train import _metrics


def _unseen_pot_test(config: dict[str, Any], features: list[str], algorithm: str) -> pd.DataFrame:
    catalog = pd.read_csv(source_path(config, "pot_catalog"), sep=";")
    pots = sorted(catalog.Cuba.astype(str).tolist())
    rng = np.random.default_rng(int(config["project"]["seed"]))
    holdout = sorted(rng.choice(pots, size=38, replace=False).tolist())
    holdout_sql = ",".join(f"'{pot}'" for pot in holdout)
    feature_sql = ",".join(f'"{name}"' for name in features)
    processed = Path(config["project"]["processed_dir"])
    scan = f"read_parquet('{as_posix(processed / 'features_2026-0[1-5].parquet')}',union_by_name=true)"
    fraction = float(config["training"]["negative_sample_fraction"])
    threshold = int(fraction * 1_000_000)
    con = connect(config)
    train = con.execute(
        f"""
        SELECT {feature_sql},target_ea_60m
        FROM {scan}
        WHERE is_valid_prediction_row=1
          AND CAST(cuba AS VARCHAR) NOT IN ({holdout_sql})
          AND (target_ea_60m=1 OR hash(cuba,timestamp,77)%1000000<{threshold})
        """
    ).fetchdf()
    con.close()
    y_train = train.pop("target_ea_60m").to_numpy(dtype=np.int8)
    x_train = train.replace([np.inf, -np.inf], np.nan)
    weights = np.where(y_train == 1, 1.0, 1.0 / fraction)
    model = make_model(algorithm, config, int(config["project"]["seed"]) + 9000)
    fit_model(model, algorithm, x_train, y_train, weights)
    joblib.dump(model, Path(config["project"]["models_dir"]) / "model_60m_unseen_pots.joblib", compress=3)

    validation = processed / "features_2026-06.parquet"
    results = {"unseen_pots": [[], []], "seen_pots": [[], []]}
    parquet = pq.ParquetFile(validation)
    columns = features + ["target_ea_60m", "is_valid_prediction_row", "cuba"]
    holdout_set = set(holdout)
    for batch in parquet.iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas()
        valid = frame.pop("is_valid_prediction_row").to_numpy(dtype=bool)
        y = frame.pop("target_ea_60m").to_numpy(dtype=np.int8)
        pots_batch = frame.pop("cuba").astype(str).to_numpy()
        x = frame.replace([np.inf, -np.inf], np.nan)
        for label, mask in (
            ("unseen_pots", valid & np.isin(pots_batch, list(holdout_set))),
            ("seen_pots", valid & ~np.isin(pots_batch, list(holdout_set))),
        ):
            if mask.any():
                results[label][0].append(y[mask])
                results[label][1].append(predict_probability(model, x.loc[mask, features]))
    rows = []
    for label, (ys, probabilities) in results.items():
        y = np.concatenate(ys)
        p = np.concatenate(probabilities)
        rows.append({"segment": label, "n_pots": 38 if label == "unseen_pots" else len(pots) - 38, **_metrics(y, p)})
    return pd.DataFrame(rows)


def _segment_metrics(config: dict[str, Any]) -> pd.DataFrame:
    reports = Path(config["project"]["reports_dir"])
    details = pd.read_csv(reports / "detalle_episodios_prediccion.csv", parse_dates=["inicio_ea"])
    details = details[
        (details.split == "test") & (details.horizon_minutes == 60) & (details.method == "direct")
    ].copy()
    catalog = pd.read_csv(source_path(config, "pot_catalog"), sep=";").rename(columns={"Cuba": "cuba"})
    catalog["cuba"] = catalog.cuba.astype(str)
    details["cuba"] = details.cuba.astype(str)
    details = details.merge(catalog, on="cuba", how="left")

    episodes_path = Path(config["project"]["interim_dir"]) / "episodios_ea.parquet"
    con = connect(config)
    history = con.execute(
        f"SELECT cuba,count(*) AS prior_ea_count FROM {parquet_scan(episodes_path)} "
        "WHERE inicio_ea<TIMESTAMP '2026-06-01' GROUP BY 1"
    ).fetchdf()
    con.close()
    history["cuba"] = history.cuba.astype(str)
    details = details.merge(history, on="cuba", how="left")
    details["prior_ea_count"] = details.prior_ea_count.fillna(0)
    details["history_band"] = pd.qcut(
        details.prior_ea_count.rank(method="first"), 3, labels=["baja", "media", "alta"]
    )

    rows = []
    for dimension, column in (("sala", "Sala"), ("grupo", "Grupo"), ("frecuencia_historica", "history_band")):
        for value, subset in details.groupby(column, dropna=False, observed=False):
            rows.append({
                "dimension": dimension,
                "segment": str(value),
                "events": len(subset),
                "episode_recall": float(subset.detected.astype(bool).mean()),
                "recall_top_5": float((subset.best_rank <= 5).mean()),
                "recall_top_10": float((subset.best_rank <= 10).mean()),
                "median_lead_min": float(subset.loc[subset.detected.astype(bool), "lead_minutes"].median()),
            })
    return pd.DataFrame(rows)


def run_robustness(config: dict[str, Any]) -> Path:
    ensure_output_dirs(config)
    models_dir = Path(config["project"]["models_dir"])
    metadata = json.loads((models_dir / "training_metadata.json").read_text(encoding="utf-8"))
    schema = json.loads((models_dir / "feature_schema.json").read_text(encoding="utf-8"))
    unseen = _unseen_pot_test(config, schema["feature_order"], metadata["selected_algorithm"])
    segments = _segment_metrics(config)
    reports = Path(config["project"]["reports_dir"])
    unseen.to_csv(reports / "robustez_cubas_no_vistas.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(reports / "robustez_segmentos.csv", index=False, encoding="utf-8-sig")
    report = f"""# Pruebas de robustez

## Generalización a cubas no vistas — validación junio, horizonte 60 min

Se reservaron determinísticamente 38 de 384 cubas y se reentrenó sin ninguna fila de esas cubas. `CUBA` e `INDPOT` no son features del modelo global.

{unseen.to_markdown(index=False)}

## Segmentos del test corregido

{segments.to_markdown(index=False)}

## Caveats

Los segmentos con pocos EA tienen alta incertidumbre. Las diferencias son diagnósticas, no evidencia causal. Julio ya no es un holdout perfectamente prístino después de la corrección de fuga v1.
"""
    output = reports / "robustez.md"
    output.write_text(report, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa cubas no vistas y segmentos.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run_robustness(load_config(args.config)))


if __name__ == "__main__":
    main()
