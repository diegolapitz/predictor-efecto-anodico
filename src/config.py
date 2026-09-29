from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Carga la configuración y agrega rutas absolutas derivadas."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    project = config["project"]
    project["root"] = str(PROJECT_ROOT)
    project["config_path"] = str(config_path)
    project["data_root"] = str((PROJECT_ROOT / project["data_root"]).resolve())
    for key in ("interim_dir", "processed_dir", "models_dir", "reports_dir", "logs_dir"):
        project[key] = str((PROJECT_ROOT / project[key]).resolve())
    return config


def ensure_output_dirs(config: dict[str, Any]) -> None:
    project = config["project"]
    for key in ("interim_dir", "processed_dir", "models_dir", "reports_dir", "logs_dir"):
        Path(project[key]).mkdir(parents=True, exist_ok=True)
    (Path(project["reports_dir"]) / "figures").mkdir(parents=True, exist_ok=True)
    (Path(project["interim_dir"]) / "duckdb_tmp").mkdir(parents=True, exist_ok=True)


def source_path(config: dict[str, Any], source_key: str) -> Path:
    return Path(config["project"]["data_root"]) / config["sources"][source_key]


def as_posix(path: str | Path) -> str:
    return Path(path).resolve().as_posix().replace("'", "''")

