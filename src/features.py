from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import as_posix, source_path
from .io import parquet_scan
from .labels import target_expression
from .semiturn import SEMITURN_END_OFFSET_HOURS, semiturn_end_case


VITM_FIELDS = [
    "POTSTATE",
    "AGEBSQ",
    "TBD",
    "RTH",
    "HBD",
    "HBC",
    "HMD",
    "HMC",
    "ALF3D",
    "NDAL2O3",
    "WRMI",
    "SMRWFC",
    "V_ACD",
    "RKM",
    "RRM",
    "POTAGE",
]



RAW_EXCLUDED_FROM_MODEL = {
    "timestamp",
    "cuba",
    "indpot",
    "is_valid_prediction_row",
    "invalid_current_signal",
    "in_or_post_ea",
    "recent_severe_gap",
    "next_target_source_60m",
    "target_ea_15m",
    "target_ea_30m",
    "target_ea_60m",
}


def _parse_month(month: str) -> tuple[datetime, datetime]:
    start = datetime.strptime(month, "%Y-%m")
    if start.month == 12:
        end = datetime(start.year + 1, 1, 1)
    else:
        end = datetime(start.year, start.month + 1, 1)
    return start, end


def _numeric_text(column: str) -> str:
    return f"try_cast(replace(nullif(trim({column}), ''), ',', '.') AS DOUBLE)"


def build_feature_query(config: dict[str, Any], month: str, episodes_path: Path, pots: list[str] | None = None) -> str:
    """Genera SQL temporal: todo feature en t usa exclusivamente valores <= t."""
    start, end = _parse_month(month)
    lookback = int(config["features"]["lookback_hours"])
    scan_start = start - timedelta(hours=lookback)
    severe_gap_s = int(config["features"]["severe_gap_minutes"]) * 60
    post_block = int(config["events"]["post_ea_block_minutes"])
    vitm_max_age = int(config["features"]["vitm_max_age_hours"])
    horizons = [int(value) for value in config["features"]["target_horizons_minutes"]]
    pot_filter = ''
    if pots is not None:
        if not pots or any(not str(p).isdecimal() for p in pots):
            raise ValueError('Pot batches require nonempty numeric catalog IDs')
        pot_filter = ' AND CAST(CUBA AS VARCHAR) IN (' + ','.join("'"+str(p)+"'" for p in pots) + ')'

    telemetry = parquet_scan(source_path(config, "telemetry_glob"), union_by_name=True)
    vitm = parquet_scan(source_path(config, "vitm"))
    anodes = parquet_scan(source_path(config, "anode_changes"))
    episodes = parquet_scan(episodes_path)
    pot_catalog = as_posix(source_path(config, "pot_catalog"))

    continuous = list(config["features"]["raw_continuous"])
    flags = list(config["features"]["raw_flags"])
    raw_select: list[str] = []
    for column in continuous:
        if column == "PAL2O3FC":
            raw_select.append(
                "CASE WHEN PAL2O3FC >= 9999 THEN NULL ELSE PAL2O3FC END AS PAL2O3FC"
            )
            raw_select.append("CASE WHEN PAL2O3FC >= 9999 THEN 1 ELSE 0 END AS pal2o3_sentinel")
        else:
            raw_select.append(column)
    raw_select.extend(flags)
    raw_select.extend(["LINE_SD", "RESTART_AFTER_SD"])

    lag_signals = ["RMFC", "UMFC", "SMRWFC", "TALIMFC"]
    lag_periods = {15: 3, 30: 6, 60: 12}
    lag_exprs: list[str] = [
        "lag(ts) OVER (PARTITION BY cuba ORDER BY ts) AS previous_ts"
    ]
    for minutes, periods in lag_periods.items():
        lag_exprs.append(
            f"lag(ts, {periods}) OVER (PARTITION BY cuba ORDER BY ts) AS ts_lag_{minutes}m"
        )
        for signal in lag_signals:
            lag_exprs.append(
                f"lag({signal}, {periods}) OVER (PARTITION BY cuba ORDER BY ts) "
                f"AS {signal.lower()}_lag_{minutes}m"
            )

    window_durations = [15, 30, 60, 120, 480]
    window_aliases = ",\n".join(
        f"w{minutes} AS (PARTITION BY cuba ORDER BY ts "
        f"RANGE BETWEEN INTERVAL {minutes} MINUTE PRECEDING AND CURRENT ROW)"
        for minutes in window_durations
    )
    rolling_exprs: list[str] = []
    for minutes in window_durations:
        rolling_exprs.extend(
            [
                f"avg(RMFC) OVER w{minutes} AS rmfc_mean_{minutes}m",
                f"stddev_samp(RMFC) OVER w{minutes} AS rmfc_std_{minutes}m",
                f"max(RMFC) OVER w{minutes} - min(RMFC) OVER w{minutes} AS rmfc_range_{minutes}m",
            ]
        )
    rolling_exprs.extend(
        [
            "min(RMFC) OVER w120 AS rmfc_min_120m",
            "avg(UMFC) OVER w30 AS umfc_mean_30m",
            "max(UMFC) OVER w30 AS umfc_max_30m",
            "avg(UMFC) OVER w60 AS umfc_mean_60m",
            "max(UMFC) OVER w60 AS umfc_max_60m",
            "avg(IMFC) OVER w60 AS imfc_mean_60m",
            "stddev_samp(IMFC) OVER w60 AS imfc_std_60m",
            "avg(SMRWFC) OVER w30 AS smrwfc_mean_30m",
            "max(SMRWFC) OVER w30 AS smrwfc_max_30m",
            "avg(SMRWFC) OVER w60 AS smrwfc_mean_60m",
            "max(SMRWFC) OVER w60 AS smrwfc_max_60m",
            "avg(WRMFC) OVER w60 AS wrmfc_mean_60m",
            "avg(TALIMFC) OVER w30 AS talimfc_mean_30m",
            "min(TALIMFC) OVER w30 AS talimfc_min_30m",
            "avg(TALIMFC) OVER w60 AS talimfc_mean_60m",
            "min(TALIMFC) OVER w60 AS talimfc_min_60m",
            "avg(PAL2O3FC) OVER w30 AS pal2o3fc_mean_30m",
            "avg(PAL2O3FC) OVER w60 AS pal2o3fc_mean_60m",
            "sum(coalesce(FEEDER_INT, 0)) OVER w60 AS feeder_int_count_60m",
            "sum(coalesce(MILD_INSTABILITY, 0) + coalesce(SEVER_INSTABILITY, 0) + coalesce(EXTD_INSTABILITY, 0)) OVER w60 AS instability_count_60m",
            "sum(coalesce(ANODE_CHANGE, 0)) OVER w120 AS anode_change_count_120m",
            "sum(coalesce(TAPPING, 0)) OVER w120 AS tapping_count_120m",
        ]
    )
    rolling_exprs.append("max(gap_s) OVER w120 AS max_gap_120m_s")

    difference_exprs: list[str] = []
    for minutes in (15, 30, 60):
        difference_exprs.extend(
            [
                f"date_diff('second', ts_lag_{minutes}m, ts) AS lag_age_{minutes}m_s",
                f"RMFC - rmfc_lag_{minutes}m AS rmfc_diff_{minutes}m",
                f"UMFC - umfc_lag_{minutes}m AS umfc_diff_{minutes}m",
                f"SMRWFC - smrwfc_lag_{minutes}m AS smrwfc_diff_{minutes}m",
                f"TALIMFC - talimfc_lag_{minutes}m AS talimfc_diff_{minutes}m",
            ]
        )

    internal_lag_columns = [
        f"{signal.lower()}_lag_{minutes}m"
        for minutes in lag_periods
        for signal in lag_signals
    ]
    internal_lag_columns.extend(f"ts_lag_{minutes}m" for minutes in lag_periods)
    internal_lag_exclude = ", ".join(internal_lag_columns)

    vitm_select = ",\n".join(
        f"{_numeric_text(field)} AS vitm_{field.lower()}" for field in VITM_FIELDS
    )
    vitm_output = ",\n".join(
        f"CASE WHEN date_diff('hour', v.vitm_ts, w.ts) BETWEEN 0 AND {vitm_max_age} "
        f"THEN v.vitm_{field.lower()} ELSE NULL END AS vitm_{field.lower()}"
        for field in VITM_FIELDS
    )
    target_output = ",\n".join(
        f"{target_expression(horizon, 'next_ea_start')} AS target_ea_{horizon}m"
        for horizon in horizons
    )
    vitm_availability = semiturn_end_case("TU_SEMI_TURNO")
    anode_availability = semiturn_end_case("TU_SEMITURNO")

    query = f"""
    WITH telemetry_raw AS (
        SELECT
            CAST(CUBA AS VARCHAR) AS cuba,
            try_cast(INDPOT AS BIGINT) AS indpot,
            DH5MN AS ts,
            {', '.join(raw_select)}
        FROM {telemetry}
        WHERE DH5MN >= TIMESTAMP '{scan_start}'
          AND DH5MN < TIMESTAMP '{end}'
          {pot_filter}
    ),
    lagged AS (
        SELECT *, {', '.join(lag_exprs)}
        FROM telemetry_raw
    ),
    with_gaps AS (
        SELECT *, date_diff('second', previous_ts, ts) AS gap_s
        FROM lagged
    ),
    windowed AS (
        SELECT *, {', '.join(rolling_exprs)}
        FROM with_gaps
        WINDOW {window_aliases}
    ),
    engineered AS (
        SELECT *,
            {', '.join(difference_exprs)},
            RMFC - rmfc_mean_60m AS rmfc_distance_mean_60m,
            CASE WHEN rmfc_std_60m > 0 THEN (RMFC - rmfc_mean_60m) / rmfc_std_60m END AS rmfc_z_60m,
            RMFC - rmfc_min_120m AS rmfc_distance_min_120m,
            sin(2 * pi() * extract('hour' FROM ts) / 24.0) AS hour_sin,
            cos(2 * pi() * extract('hour' FROM ts) / 24.0) AS hour_cos,
            sin(2 * pi() * extract('dow' FROM ts) / 7.0) AS weekday_sin,
            cos(2 * pi() * extract('dow' FROM ts) / 7.0) AS weekday_cos
        FROM windowed
    ),
    valid_pots AS (
        SELECT CAST(Cuba AS VARCHAR) AS cuba
        FROM read_csv('{pot_catalog}', delim=';', header=true)
    ),
    vitm_normalized AS (
        SELECT
            CAST(CUBA AS VARCHAR) AS cuba,
            strptime(FT_TURNO, '%d/%m/%Y %H:%M:%S')
                + {vitm_availability} AS vitm_ts,
            {vitm_select}
        FROM {vitm}
        WHERE CAST(CUBA AS VARCHAR) IN (SELECT cuba FROM valid_pots)
          AND strptime(FT_TURNO, '%d/%m/%Y %H:%M:%S') >= TIMESTAMP '{scan_start - timedelta(hours=vitm_max_age + 24)}'
          AND strptime(FT_TURNO, '%d/%m/%Y %H:%M:%S') < TIMESTAMP '{end}'
    ),
    anode_change_times AS (
        SELECT DISTINCT
            CAST(CUBA AS VARCHAR) AS cuba,
            FT_FECHA
                + {anode_availability} AS anode_change_ts
        FROM {anodes}
        WHERE FT_FECHA >= TIMESTAMP '{start - timedelta(days=60)}'
          AND FT_FECHA < TIMESTAMP '{end}'
    ),
    with_vitm AS (
        SELECT
            w.*,
            v.vitm_ts,
            date_diff('hour', v.vitm_ts, w.ts) AS vitm_age_hours,
            CASE WHEN v.vitm_ts IS NULL OR date_diff('hour', v.vitm_ts, w.ts) NOT BETWEEN 0 AND {vitm_max_age}
                 THEN 1 ELSE 0 END AS vitm_missing_or_stale,
            {vitm_output}
        FROM engineered w
        ASOF LEFT JOIN vitm_normalized v
          ON w.cuba = v.cuba AND w.ts >= v.vitm_ts
    ),
    context AS (
        SELECT
            wv.*,
            date_diff('hour', a.anode_change_ts, wv.ts) AS hours_since_anode_change
        FROM with_vitm wv
        ASOF LEFT JOIN anode_change_times a
          ON wv.cuba = a.cuba AND wv.ts >= a.anode_change_ts
    ),
    joined_events AS (
        SELECT
            c.*,
            previous_ea.inicio_ea AS previous_ea_start,
            previous_ea.fin_ea AS previous_ea_end,
            next_ea.inicio_ea AS next_ea_start,
            next_ea.source AS next_ea_source
        FROM context c
        ASOF LEFT JOIN {episodes} previous_ea
          ON c.cuba = previous_ea.cuba AND c.ts >= previous_ea.inicio_ea
        ASOF LEFT JOIN {episodes} next_ea
          ON c.cuba = next_ea.cuba AND c.ts < next_ea.inicio_ea
    ),
    c AS (
        SELECT *,
            date_diff('minute', previous_ea_start, ts) AS minutes_since_previous_ea,
            CASE WHEN previous_ea_start IS NOT NULL
                       AND ts <= previous_ea_end + INTERVAL {post_block} MINUTE
                 THEN 1 ELSE 0 END AS in_or_post_ea,
            CASE WHEN coalesce(max_gap_120m_s, gap_s, 0) > {severe_gap_s} THEN 1 ELSE 0 END AS recent_severe_gap,
            CASE WHEN coalesce(LINE_SD, 0) <> 0 OR IMFC IS NULL OR UMFC IS NULL OR RMFC IS NULL
                 THEN 1 ELSE 0 END AS invalid_current_signal
        FROM joined_events
    )
    SELECT
        ts AS timestamp,
        cuba,
        indpot,
        * EXCLUDE (ts, cuba, indpot, previous_ts, previous_ea_start, previous_ea_end,
                   next_ea_start, next_ea_source, vitm_ts,
                   {internal_lag_exclude}),
        CASE WHEN in_or_post_ea = 0 AND recent_severe_gap = 0 AND invalid_current_signal = 0
             THEN 1 ELSE 0 END AS is_valid_prediction_row,
        CASE WHEN next_ea_start <= ts + INTERVAL 60 MINUTE THEN next_ea_source END AS next_target_source_60m,
        {target_output}
    FROM c
    WHERE ts >= TIMESTAMP '{start}' AND ts < TIMESTAMP '{end}'
    ORDER BY timestamp, cuba
    """
    return query


def model_feature_columns(all_columns: list[str]) -> list[str]:
    """Excluye identificadores, targets y variables operativas que contienen futuro."""
    return [
        column
        for column in all_columns
        if column not in RAW_EXCLUDED_FROM_MODEL
        and not column.startswith("target_ea_")
        and column not in {"gap_s", "LINE_SD", "RESTART_AFTER_SD"}
    ]
