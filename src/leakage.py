from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .config import ensure_output_dirs, load_config
from .features import VITM_FIELDS, model_feature_columns
from .io import connect, parquet_scan


FORBIDDEN_COLUMNS = {
    "ANODE_EFFECT",
    "FH_DHEVT",
    "N_NUMEVT",
    "V_ASSVAL1",
    "V_ASSVAL2",
    "vitm_ntea",
    "vitm_dtea",
    "vitm_sea",
    "next_ea_start",
    "previous_ea_end",
}


def run_leakage_audit(config: dict[str, Any]) -> Path:
    ensure_output_dirs(config)
    processed = Path(config["project"]["processed_dir"])
    files = sorted(processed.glob("features_2026-*.parquet"))
    if not files:
        raise FileNotFoundError("No hay particiones de features; ejecutar src.build_dataset primero.")

    schemas = [pq.ParquetFile(path).schema_arrow.names for path in files]
    schema_equal = all(columns == schemas[0] for columns in schemas[1:])
    all_columns = list(schemas[0])
    features = model_feature_columns(all_columns)
    forbidden_present = sorted(FORBIDDEN_COLUMNS.intersection(features))

    scan = parquet_scan(processed / "features_2026-*.parquet", union_by_name=True)
    con = connect(config)
    vitm_non_null = " OR ".join(f"vitm_{field.lower()} IS NOT NULL" for field in VITM_FIELDS)
    checks = con.execute(
        f"""
        SELECT
            count(*) AS n_rows,
            sum(CASE WHEN target_ea_15m > target_ea_30m OR target_ea_30m > target_ea_60m THEN 1 ELSE 0 END) AS target_order_violations,
            sum(CASE WHEN hours_since_anode_change < 0 THEN 1 ELSE 0 END) AS future_anode_join_rows,
            sum(CASE WHEN vitm_age_hours < 0 THEN 1 ELSE 0 END) AS future_vitm_join_rows,
            sum(CASE WHEN vitm_missing_or_stale = 1 AND ({vitm_non_null}) THEN 1 ELSE 0 END) AS stale_vitm_value_rows,
            sum(CASE WHEN is_valid_prediction_row = 1 AND (in_or_post_ea = 1 OR recent_severe_gap = 1 OR invalid_current_signal = 1) THEN 1 ELSE 0 END) AS invalid_rows_marked_valid
        FROM {scan}
        """
    ).fetchdf().iloc[0]
    con.close()

    status = "PASS" if (
        schema_equal
        and not forbidden_present
        and int(checks.target_order_violations) == 0
        and int(checks.future_anode_join_rows) == 0
        and int(checks.future_vitm_join_rows) == 0
        and int(checks.stale_vitm_value_rows) == 0
        and int(checks.invalid_rows_marked_valid) == 0
    ) else "FAIL"

    report = f"""# Auditoría de fuga temporal

## Resultado: {status}

Se revisaron {int(checks.n_rows):,} filas y {len(features)} features entrenables en {len(files)} particiones mensuales.

## Controles ejecutados

| Control | Resultado |
|---|---:|
| Esquema idéntico entre particiones | {schema_equal} |
| Columnas prohibidas presentes en features | {', '.join(forbidden_present) if forbidden_present else 'ninguna'} |
| Violaciones `target_15m ≤ target_30m ≤ target_60m` | {int(checks.target_order_violations)} |
| As-of de cambios anódicos desde el futuro | {int(checks.future_anode_join_rows)} |
| As-of de VITM desde el futuro | {int(checks.future_vitm_join_rows)} |
| Valores VITM conservados pese a antigüedad > límite | {int(checks.stale_vitm_value_rows)} |
| Filas inválidas marcadas como válidas | {int(checks.invalid_rows_marked_valid)} |

## Reglas de construcción verificadas

- Los targets se definen con un evento en `(t, t+h]`; el instante `t` no cuenta.
- Todas las ventanas eléctricas son `RANGE ... PRECEDING AND CURRENT ROW`, nunca centradas.
- VITM y cambios anódicos se unen con `ASOF` hacia atrás.
- Los agregados por semiturno se consideran disponibles al cierre, nunca al inicio.
- `VITM.NTEA`, `DTEA` y `SEA` se excluyen por ser resultados del propio EA.
- VITM se anula cuando su edad supera {int(config['features']['vitm_max_age_hours'])} h y conserva un indicador de faltante/obsolescencia.
- Se excluyen del conjunto entrenable el evento siguiente, `ANODE_EFFECT`, códigos 241/242, tensión final y duración final.
- La normalización/imputación de cada modelo se ajusta sólo con train; junio selecciona modelo/umbral y julio queda bloqueado hasta `src.evaluate`.

## Caveat no resoluble con código

El export oficial `THISEVT` no cubre enero-marzo ni julio. En esos períodos el target es un proxy validado y marcado. Esto es incertidumbre de etiqueta, no fuga, pero impide homologar las métricas finales como evaluación 100% oficial.
"""
    output = Path(config["project"]["reports_dir"]) / "auditoria_fugas.md"
    output.write_text(report, encoding="utf-8")
    if status != "PASS":
        raise RuntimeError(f"La auditoría de fuga temporal falló. Revisar {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita fuga temporal del dataset procesado.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(run_leakage_audit(load_config(args.config)))


if __name__ == "__main__":
    main()
