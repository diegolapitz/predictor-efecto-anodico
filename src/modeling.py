from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(algorithm: str, config: dict[str, Any], seed: int):
    if algorithm == "lightgbm":
        params = dict(config["training"]["lightgbm"])
        return LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=int(config["runtime"]["duckdb_threads"]),
            verbosity=-1,
            **params,
        )
    if algorithm == "logistic_regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=250,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if algorithm == "extra_trees":
        params = dict(config["training"]["extra_trees"])
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        random_state=seed,
                        n_jobs=int(config["runtime"]["duckdb_threads"]),
                        class_weight=None,
                        **params,
                    ),
                ),
            ]
        )
    raise ValueError(f"Algoritmo desconocido: {algorithm}")


def fit_model(model, algorithm: str, x, y, sample_weight):
    if algorithm in {"logistic_regression", "extra_trees"}:
        model.fit(x, y, model__sample_weight=sample_weight)
    else:
        model.fit(x, y, sample_weight=sample_weight)
    return model


def predict_probability(model, x) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def feature_groups(all_features: list[str]) -> dict[str, list[str]]:
    resistance = [
        name for name in all_features
        if "rmfc" in name.lower() or name in {"RMFC", "RKFC", "RTHFC", "WRMFC"}
    ]
    feeding_terms = ("talim", "pal2o3", "feed", "al2o3")
    feeding = sorted(set(resistance + [
        name for name in all_features if any(term in name.lower() for term in feeding_terms)
    ]), key=all_features.index)
    event_terms = (
        "instability", "anode", "tapping", "previous_ea", "beam", "interrupt",
        "hour_", "weekday_", "smrwfc", "wrmfc",
    )
    events = sorted(set(feeding + [
        name for name in all_features if any(term in name.lower() for term in event_terms)
    ]), key=all_features.index)
    return {
        "solo_resistencia": resistance,
        "resistencia_alimentacion": feeding,
        "resistencia_alimentacion_eventos": events,
        "todas_incluye_vitm": list(all_features),
    }

