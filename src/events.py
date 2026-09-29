from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import as_posix, ensure_output_dirs, load_config, source_path
from .io import connect, get_logger, parquet_scan


def build_episodes(config: dict[str, Any], force: bool = False) -> Path:
    """Construye episodios híbridos: oficiales en cobertura y proxy fuera de ella."""
    ensure_output_dirs(config)
    logger = get_logger("events", config)
    output = Path(config["project"]["interim_dir"]) / "episodios_ea.parquet"
    metadata_path = Path(config["project"]["interim_dir"]) / "episodios_ea_metadata.json"
    if output.exists() and not force:
        logger.info("Reutilizando episodios existentes: %s", output)
        return output

    events_cfg = config["events"]
    event_scan = parquet_scan(source_path(config, "official_events"))
    telemetry_scan = parquet_scan(source_path(config, "telemetry_glob"), union_by_name=True)
    start_code = int(events_cfg["official_start_code"])
    end_code = int(events_cfg["official_end_code"])
    offset = int(events_cfg["proxy_time_offset_seconds"])
    proxy_duration = int(events_cfg["proxy_default_duration_seconds"])
    uncertainty = int(events_cfg["proxy_timestamp_uncertainty_seconds"])

    con = connect(config)
    coverage = con.execute(
        f"SELECT min(FH_DHEVT), max(FH_DHEVT), count(*) FROM {event_scan}"
    ).fetchone()
    if not coverage or coverage[0] is None:
        raise RuntimeError("El archivo THISEVT no contiene timestamps utilizables.")
    coverage_start, coverage_end, coverage_rows = coverage

    official_cte = f"""
    official_starts AS (
        SELECT
            CAST(CUBA AS VARCHAR) AS cuba,
            ID_HISEVT AS start_event_id,
            FH_DHEVT AS inicio_ea,
            row_number() OVER (
                PARTITION BY CAST(CUBA AS VARCHAR) ORDER BY FH_DHEVT, ID_HISEVT
            ) AS episode_number
        FROM {event_scan}
        WHERE N_NUMEVT = {start_code}
    ),
    official_ends AS (
        SELECT
            CAST(CUBA AS VARCHAR) AS cuba,
            ID_HISEVT AS end_event_id,
            FH_DHEVT AS fin_ea,
            try_cast(V_ASSVAL1 AS DOUBLE) AS tension_fin_v,
            try_cast(V_ASSVAL2 AS DOUBLE) AS duracion_reportada_s,
            row_number() OVER (
                PARTITION BY CAST(CUBA AS VARCHAR) ORDER BY FH_DHEVT, ID_HISEVT
            ) AS episode_number
        FROM {event_scan}
        WHERE N_NUMEVT = {end_code}
    ),
    official AS (
        SELECT
            s.cuba,
            s.inicio_ea,
            e.fin_ea,
            e.duracion_reportada_s,
            date_diff('second', s.inicio_ea, e.fin_ea)::DOUBLE AS duracion_entre_eventos_s,
            e.tension_fin_v,
            'thisevt_oficial'::VARCHAR AS source,
            ('official:' || CAST(s.start_event_id AS VARCHAR))::VARCHAR AS event_id,
            s.start_event_id,
            e.end_event_id,
            NULL::TIMESTAMP AS proxy_timestamp,
            0::INTEGER AS timestamp_uncertainty_s,
            false AS fin_estimado
        FROM official_starts s
        FULL JOIN official_ends e USING (cuba, episode_number)
    )
    """

    proxy_cte = f"""
    proxy AS (
        SELECT
            CAST(CUBA AS VARCHAR) AS cuba,
            DH5MN - INTERVAL {offset} SECOND AS inicio_ea,
            DH5MN - INTERVAL {offset} SECOND + INTERVAL {proxy_duration} SECOND AS fin_ea,
            NULL::DOUBLE AS duracion_reportada_s,
            {proxy_duration}::DOUBLE AS duracion_entre_eventos_s,
            NULL::DOUBLE AS tension_fin_v,
            'his5m_anode_effect_proxy'::VARCHAR AS source,
            ('proxy:' || CAST(CUBA AS VARCHAR) || ':' || strftime(DH5MN, '%Y%m%d%H%M%S'))::VARCHAR AS event_id,
            NULL::BIGINT AS start_event_id,
            NULL::BIGINT AS end_event_id,
            DH5MN AS proxy_timestamp,
            {uncertainty}::INTEGER AS timestamp_uncertainty_s,
            true AS fin_estimado
        FROM {telemetry_scan}
        WHERE ANODE_EFFECT = 1
          AND (DH5MN < TIMESTAMP '{coverage_start}' OR DH5MN > TIMESTAMP '{coverage_end}')
    )
    """

    invalid = con.execute(
        f"""
        WITH {official_cte}
        SELECT
            count(*) FILTER (WHERE inicio_ea IS NULL) AS missing_start,
            count(*) FILTER (WHERE fin_ea IS NULL) AS missing_end,
            count(*) FILTER (WHERE fin_ea < inicio_ea) AS end_before_start
        FROM official
        """
    ).fetchone()
    if invalid != (0, 0, 0):
        raise RuntimeError(f"No se pudieron emparejar limpiamente los eventos EA: {invalid}")

    combined = f"""
        WITH {official_cte}, {proxy_cte}
        SELECT * FROM official
        UNION ALL BY NAME
        SELECT * FROM proxy
        ORDER BY cuba, inicio_ea
    """
    con.execute(
        f"COPY ({combined}) TO '{as_posix(output)}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    summary = con.execute(
        f"SELECT source, count(*) AS episodes, count(DISTINCT cuba) AS cubas, "
        f"min(inicio_ea) AS min_ts, max(inicio_ea) AS max_ts FROM {parquet_scan(output)} "
        "GROUP BY 1 ORDER BY 1"
    ).fetchdf()
    metadata = {
        "official_event_coverage_start": str(coverage_start),
        "official_event_coverage_end": str(coverage_end),
        "official_event_rows_all_codes": int(coverage_rows),
        "start_code": start_code,
        "end_code": end_code,
        "proxy_time_offset_seconds": offset,
        "proxy_default_duration_seconds": proxy_duration,
        "proxy_timestamp_uncertainty_seconds": uncertainty,
        "sources": summary.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Episodios escritos en %s", output)
    logger.info("\n%s", summary.to_string(index=False))
    con.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye episodios de efecto anódico.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_episodes(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()

