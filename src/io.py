from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import duckdb

from .config import as_posix, ensure_output_dirs


def get_logger(name: str, config: dict[str, Any]) -> logging.Logger:
    ensure_output_dirs(config)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        Path(config["project"]["logs_dir"]) / f"{name}.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def connect(config: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    ensure_output_dirs(config)
    connection = duckdb.connect()
    runtime = config["runtime"]
    connection.execute(f"SET threads={int(runtime['duckdb_threads'])}")
    connection.execute(f"SET memory_limit='{runtime['duckdb_memory_limit']}'")
    temp_dir = as_posix(Path(config["project"]["interim_dir"]) / "duckdb_tmp")
    connection.execute(f"SET temp_directory='{temp_dir}'")
    connection.execute("SET preserve_insertion_order=false")
    return connection


def parquet_scan(path: str | Path, union_by_name: bool = False) -> str:
    union = ", union_by_name=true" if union_by_name else ""
    return f"read_parquet('{as_posix(path)}'{union})"

