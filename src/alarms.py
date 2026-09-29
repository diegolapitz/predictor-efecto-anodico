from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import as_posix
from .io import connect, parquet_scan


def evaluate_operational_predictions(
    config: dict[str, Any],
    predictions_path: Path,
    episodes_path: Path,
    split_name: str,
    period_start: str,
    period_end: str,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalúa alarmas directas, persistencia 2/3, cooldown y ranking top-k."""
    con = connect(config)
    pred = parquet_scan(predictions_path)
    episodes = parquet_scan(episodes_path)
    persistence_window = int(config["alarms"]["persistence_window"])
    persistence_required = int(config["alarms"]["persistence_required"])
    cooldown = int(config["alarms"]["cooldown_minutes"])
    top_k = [int(value) for value in config["alarms"]["ranking_k"]]
    horizons = [int(value) for value in config["features"]["target_horizons_minutes"]]

    metrics_rows: list[dict[str, Any]] = []
    event_frames: list[pd.DataFrame] = []
    alarm_frames: list[pd.DataFrame] = []

    cuba_days = int(
        con.execute(
            f"SELECT count(DISTINCT cuba || ':' || CAST(CAST(timestamp AS DATE) AS VARCHAR)) "
            f"FROM {pred} WHERE is_valid_prediction_row=1"
        ).fetchone()[0]
    )
    calendar_days = int(
        con.execute(
            f"SELECT count(DISTINCT CAST(timestamp AS DATE)) FROM {pred}"
        ).fetchone()[0]
    )

    for horizon in horizons:
        threshold = float(thresholds[str(horizon)])
        risk = f"risk_{horizon}m"
        con.execute("DROP TABLE IF EXISTS scored")
        con.execute(
            f"""
            CREATE TEMP TABLE scored AS
            WITH base AS (
                SELECT
                    timestamp,
                    CAST(cuba AS VARCHAR) AS cuba,
                    {risk} AS risk,
                    time_bucket(INTERVAL '5 minutes', timestamp) AS bucket_5m,
                    CASE WHEN {risk} >= {threshold} THEN 1 ELSE 0 END AS direct_alarm
                FROM {pred}
                WHERE is_valid_prediction_row = 1
            ), ranked AS (
                SELECT *,
                    row_number() OVER (
                        PARTITION BY bucket_5m ORDER BY risk DESC, cuba
                    ) AS ranking,
                    sum(direct_alarm) OVER (
                        PARTITION BY cuba ORDER BY timestamp
                        ROWS BETWEEN {persistence_window - 1} PRECEDING AND CURRENT ROW
                    ) AS persistence_hits
                FROM base
            )
            SELECT *, CASE WHEN persistence_hits >= {persistence_required} THEN 1 ELSE 0 END AS persistent_alarm
            FROM ranked
            """
        )

        event_rank = con.execute(
            f"""
            WITH period_events AS (
                SELECT event_id, cuba, inicio_ea, source
                FROM {episodes}
                WHERE inicio_ea >= TIMESTAMP '{period_start}'
                  AND inicio_ea < TIMESTAMP '{period_end}'
            )
            SELECT
                e.event_id, e.cuba, e.inicio_ea, e.source,
                min(s.ranking) AS best_rank
            FROM period_events e
            LEFT JOIN scored s
              ON e.cuba = s.cuba
             AND s.timestamp >= e.inicio_ea - INTERVAL {horizon} MINUTE
             AND s.timestamp < e.inicio_ea
            GROUP BY 1,2,3,4
            """
        ).fetchdf()

        for method_name, alarm_column in (
            ("direct", "direct_alarm"),
            (f"persistent_{persistence_required}of{persistence_window}_cooldown_{cooldown}m", "persistent_alarm"),
        ):
            con.execute("DROP TABLE IF EXISTS alarm_points")
            con.execute(
                f"CREATE TEMP TABLE alarm_points AS "
                f"SELECT timestamp,cuba,bucket_5m,ranking,risk FROM scored WHERE {alarm_column}=1"
            )
            event_match = con.execute(
                f"""
                WITH period_events AS (
                    SELECT event_id, cuba, inicio_ea, source
                    FROM {episodes}
                    WHERE inicio_ea >= TIMESTAMP '{period_start}'
                      AND inicio_ea < TIMESTAMP '{period_end}'
                )
                SELECT
                    e.event_id, e.cuba, e.inicio_ea, e.source,
                    max(date_diff('second', a.timestamp, e.inicio_ea) / 60.0) AS lead_minutes,
                    count(a.timestamp) AS alarm_rows_in_window
                FROM period_events e
                LEFT JOIN alarm_points a
                  ON e.cuba = a.cuba
                 AND a.timestamp >= e.inicio_ea - INTERVAL {horizon} MINUTE
                 AND a.timestamp < e.inicio_ea
                GROUP BY 1,2,3,4
                """
            ).fetchdf()
            event_match = event_match.merge(event_rank, on=["event_id", "cuba", "inicio_ea", "source"], how="left")
            event_match["detected"] = event_match["alarm_rows_in_window"] > 0
            event_match["split"] = split_name
            event_match["horizon_minutes"] = horizon
            event_match["method"] = method_name
            event_frames.append(event_match)

            alarm_episodes = con.execute(
                f"""
                WITH marked AS (
                    SELECT *,
                        lag(timestamp) OVER (PARTITION BY cuba ORDER BY timestamp) AS previous_alarm_ts
                    FROM alarm_points
                ), grouped AS (
                    SELECT *,
                        sum(CASE WHEN previous_alarm_ts IS NULL
                                      OR timestamp > previous_alarm_ts + INTERVAL {cooldown} MINUTE
                                 THEN 1 ELSE 0 END)
                        OVER (PARTITION BY cuba ORDER BY timestamp) AS alarm_group
                    FROM marked
                ), alarm_episode AS (
                    SELECT cuba, alarm_group,
                           min(timestamp) AS alarm_start,
                           max(timestamp) AS alarm_end,
                           count(*) AS alarm_rows,
                           max(risk) AS max_risk
                    FROM grouped
                    GROUP BY 1,2
                ), period_events AS (
                    SELECT event_id, cuba, inicio_ea
                    FROM {episodes}
                    WHERE inicio_ea >= TIMESTAMP '{period_start}'
                      AND inicio_ea < TIMESTAMP '{period_end}'
                )
                SELECT
                    a.*,
                    date_diff('second', alarm_start, alarm_end) / 60.0 + 5.0 AS duration_minutes,
                    count(DISTINCT e.event_id) > 0 AS true_alarm
                FROM alarm_episode a
                LEFT JOIN period_events e
                  ON a.cuba = e.cuba
                 AND e.inicio_ea > a.alarm_start
                 AND e.inicio_ea <= a.alarm_end + INTERVAL {horizon} MINUTE
                GROUP BY ALL
                """
            ).fetchdf()
            alarm_episodes["split"] = split_name
            alarm_episodes["horizon_minutes"] = horizon
            alarm_episodes["method"] = method_name
            alarm_frames.append(alarm_episodes)

            simultaneous = con.execute(
                "SELECT median(n_pots), max(n_pots), avg(n_pots) FROM "
                "(SELECT bucket_5m, count(DISTINCT cuba) AS n_pots FROM alarm_points GROUP BY 1)"
            ).fetchone()

            for event_source in ["all"] + sorted(event_match["source"].dropna().unique().tolist()):
                subset = event_match if event_source == "all" else event_match[event_match.source == event_source]
                n_events = len(subset)
                detected = int(subset.detected.sum())
                row: dict[str, Any] = {
                    "split": split_name,
                    "horizon_minutes": horizon,
                    "method": method_name,
                    "event_source": event_source,
                    "events": n_events,
                    "detected_events": detected,
                    "episode_recall": detected / n_events if n_events else math.nan,
                    "lead_median_min": float(subset.loc[subset.detected, "lead_minutes"].median()) if detected else math.nan,
                }
                for lead in (5, 10, 15, 20, 30):
                    row[f"recall_lead_ge_{lead}m"] = (
                        float((subset.lead_minutes >= lead).sum() / n_events) if n_events else math.nan
                    )
                for k in top_k:
                    row[f"recall_top_{k}"] = (
                        float((subset.best_rank <= k).sum() / n_events) if n_events else math.nan
                    )
                if event_source == "all":
                    n_alarm_episodes = len(alarm_episodes)
                    false_alarms = int((~alarm_episodes.true_alarm.astype(bool)).sum())
                    row.update(
                        {
                            "alarm_episodes": n_alarm_episodes,
                            "false_alarm_episodes": false_alarms,
                            "false_alarms_per_cuba_day": false_alarms / cuba_days if cuba_days else math.nan,
                            "alarms_per_day": n_alarm_episodes / calendar_days if calendar_days else math.nan,
                            "alarm_episode_precision": float(alarm_episodes.true_alarm.mean()) if n_alarm_episodes else math.nan,
                            "alarm_duration_median_min": float(alarm_episodes.duration_minutes.median()) if n_alarm_episodes else math.nan,
                            "simultaneous_pots_median": float(simultaneous[0]) if simultaneous[0] is not None else 0.0,
                            "simultaneous_pots_max": int(simultaneous[1]) if simultaneous[1] is not None else 0,
                            "simultaneous_pots_mean": float(simultaneous[2]) if simultaneous[2] is not None else 0.0,
                        }
                    )
                metrics_rows.append(row)

    con.close()
    return pd.DataFrame(metrics_rows), pd.concat(event_frames, ignore_index=True), pd.concat(alarm_frames, ignore_index=True)
