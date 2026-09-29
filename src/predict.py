from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import as_posix, ensure_output_dirs, load_config
from .io import connect
from .modeling import predict_probability


def score_and_rank(
    config: dict[str, Any], input_path: Path, output_path: Path
) -> Path:
    """Produce riesgos y rankings por bloque de 5 minutos para filas válidas."""
    if (Path(config['project']['processed_dir'])/'ST6_INVALIDATED.json').exists():
        raise RuntimeError('ST6: scoring blocked while feature/model reconstruction is incomplete.')
    ensure_output_dirs(config)
    models_dir = Path(config["project"]["models_dir"])
    schema = json.loads((models_dir / "feature_schema.json").read_text(encoding="utf-8"))
    features = schema["feature_order"]
    models = {h: joblib.load(models_dir / f"model_{h}m.joblib") for h in (15, 30, 60)}
    score_path = Path(config["project"]["interim_dir"]) / f"{output_path.stem}_scores.tmp.parquet"
    if score_path.exists():
        score_path.unlink()

    parquet = pq.ParquetFile(input_path)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in parquet.iter_batches(
            batch_size=100_000,
            columns=list(dict.fromkeys(["timestamp", "cuba", "is_valid_prediction_row"] + features)),
        ):
            frame = batch.to_pandas()
            valid = frame["is_valid_prediction_row"].to_numpy(dtype=bool)
            if not valid.any():
                continue
            x = frame.loc[valid, features].replace([np.inf, -np.inf], np.nan)
            result = frame.loc[valid, ["timestamp", "cuba"]].copy()
            for horizon, model in models.items():
                result[f"riesgo_{horizon}m"] = predict_probability(model, x).astype(np.float32)
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(score_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("El input no contiene filas válidas para puntuar.")

    con = connect(config)
    ranked_tmp = output_path.with_suffix(".tmp.parquet")
    if ranked_tmp.exists():
        ranked_tmp.unlink()
    con.execute(
        f"""
        COPY (WITH latest_per_pot_block AS (
            SELECT
                time_bucket(INTERVAL '5 minutes', timestamp) AS timestamp,
                timestamp AS feature_timestamp,
                cuba,
                riesgo_15m,
                riesgo_30m,
                riesgo_60m
            FROM read_parquet('{as_posix(score_path)}')
            QUALIFY row_number() OVER (
                PARTITION BY time_bucket(INTERVAL '5 minutes', timestamp), cuba
                ORDER BY timestamp DESC
            ) = 1
        )
            SELECT
                timestamp,
                feature_timestamp,
                cuba,
                riesgo_15m,
                riesgo_30m,
                riesgo_60m,
                row_number() OVER (
                    PARTITION BY timestamp
                    ORDER BY riesgo_15m DESC, cuba
                ) AS ranking_15m,
                row_number() OVER (
                    PARTITION BY timestamp
                    ORDER BY riesgo_30m DESC, cuba
                ) AS ranking_30m,
                row_number() OVER (
                    PARTITION BY timestamp
                    ORDER BY riesgo_60m DESC, cuba
                ) AS ranking_60m,
                row_number() OVER (
                    PARTITION BY timestamp
                    ORDER BY riesgo_60m DESC, cuba
                ) AS ranking
            FROM latest_per_pot_block
            ORDER BY timestamp, ranking_60m
        ) TO '{as_posix(ranked_tmp)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
    )
    con.close()
    pq.ParquetFile(ranked_tmp)
    ranked_tmp.replace(output_path)
    score_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Puntúa un dataset de features y rankea cubas.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default="data/processed/features_2026-07.parquet")
    parser.add_argument("--output", default="data/processed/ranking_2026-07.parquet")
    args = parser.parse_args()
    config = load_config(args.config)
    root = Path(config["project"]["root"])
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = root / input_path
    if not output_path.is_absolute():
        output_path = root / output_path
    print(score_and_rank(config, input_path.resolve(), output_path.resolve()))


if __name__ == "__main__":
    main()
