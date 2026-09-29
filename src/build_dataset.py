from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from .config import as_posix, ensure_output_dirs, load_config, source_path
from .events import build_episodes
from .features import build_feature_query
from .io import connect, get_logger, parquet_scan


def build_month(config: dict[str, Any], month: str, episodes_path: Path, force: bool = False) -> dict[str, Any]:
    output = Path(config["project"]["processed_dir"]) / f"features_{month}.parquet"
    logger = get_logger("build_dataset", config)
    if output.exists() and not force:
        logger.info("Reutilizando %s", output.name)
    else:
        query = build_feature_query(config, month, episodes_path)
        con = connect(config)
        started = time.time()
        logger.info("Construyendo features para %s", month)
        write_path = output.with_suffix(".tmp.parquet") if force and output.exists() else output
        if write_path.exists():
            write_path.unlink()
        batch_size=int(config['runtime'].get('pot_batch_size',0))
        if batch_size:
            # Every temporal window and ASOF is partitioned by cuba. Batching
            # those independent partitions changes resource use, not history.
            with source_path(config,'pot_catalog').open(encoding='utf-8-sig') as f:
                pots=sorted({r['Cuba'] for r in csv.DictReader(f,delimiter=';')})
            parts_dir=output.parent/(output.stem+'_parts');parts_dir.mkdir(exist_ok=True)
            paths=[]
            for i in range(0,len(pots),batch_size):
                part=parts_dir/f'batch_{i//batch_size:03d}.parquet'
                # Regenerate every part, including after an interrupted attempt.
                if part.exists(): part.unlink()
                part_query=build_feature_query(config,month,episodes_path,pots=pots[i:i+batch_size])
                con.execute(f"COPY ({part_query}) TO '{as_posix(part)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)")
                paths.append(str(part))
                logger.info('Partición %s: %d/%d cubas reconstruidas',month,min(i+batch_size,len(pots)),len(pots))
            con.execute(f"COPY (SELECT * FROM read_parquet(?) ORDER BY timestamp,cuba) TO '{as_posix(write_path)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)", [paths])
        else:
            con.execute(
                f"COPY ({query}) TO '{as_posix(write_path)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
            )
        if write_path != output:
            # Verifica el footer antes de reemplazar una partición reutilizable.
            import pyarrow.parquet as pq
            pq.ParquetFile(write_path)
            write_path.replace(output)
        logger.info("Partición %s escrita en %.1f s", month, time.time() - started)
        con.close()

    con = connect(config)
    horizons = [int(value) for value in config["features"]["target_horizons_minutes"]]
    target_expr = ", ".join(
        f"sum(target_ea_{h}m) FILTER (WHERE is_valid_prediction_row=1) AS positives_{h}m"
        for h in horizons
    )
    summary = con.execute(
        f"SELECT '{month}' AS month, count(*) AS n_rows, "
        "sum(is_valid_prediction_row) AS valid_rows, "
        f"{target_expr}, "
        "sum(CASE WHEN next_target_source_60m='his5m_anode_effect_proxy' THEN 1 ELSE 0 END) AS proxy_positive_rows_60m "
        f"FROM {parquet_scan(output)}"
    ).fetchdf().iloc[0].to_dict()
    summary["path"] = str(output)
    summary["size_mb"] = round(output.stat().st_size / (1024 * 1024), 3)
    con.close()
    return summary


def build_all(config: dict[str, Any], months: list[str] | None = None, force: bool = False) -> Path:
    ensure_output_dirs(config)
    episodes_path = build_episodes(config)
    selected = months or list(config["dates"]["build_months"])
    summaries = [build_month(config, month, episodes_path, force=force) for month in selected]
    manifest = Path(config["project"]["processed_dir"]) / "dataset_manifest.json"
    manifest.write_text(json.dumps(summaries, indent=2, default=str), encoding="utf-8")
    get_logger("build_dataset", config).info("Manifest escrito en %s", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye features y targets por mes.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--month", action="append", dest="months")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pot-batch-size", type=int, default=0, help='Bound memory by independent pot partitions')
    parser.add_argument("--memory-limit", default=None)
    args = parser.parse_args()
    config=load_config(args.config)
    if args.pot_batch_size:
        if args.pot_batch_size<1: parser.error('--pot-batch-size must be positive')
        config['runtime']['pot_batch_size']=args.pot_batch_size
    if args.memory_limit: config['runtime']['duckdb_memory_limit']=args.memory_limit
    print(build_all(config, months=args.months, force=args.force))


if __name__ == "__main__":
    main()
