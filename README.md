# Predicción de efectos anódicos

Proyecto experimental de aprendizaje automático para ordenar cubas de electrólisis según el riesgo de inicio de un efecto anódico en horizontes de 15, 30 y 60 minutos. Incluye preparación de series temporales, construcción de variables, entrenamiento, evaluación y generación de rankings.

## Estado

**Investigación pausada.** Se corrigió en el código la disponibilidad temporal de los registros de semiturno, cuyo sexto cierre ocurre a las 30 horas de `FT_TURNO`. Los conjuntos procesados y modelos anteriores a esa corrección no se reconstruyeron y validaron de forma integral. Por eso, este repositorio no presenta métricas vigentes ni un sistema de alarmas listo para producción.

La evaluación debe respetar particiones temporales y medir tanto detección y precisión como la carga de alarmas. Los resultados retrospectivos no prueban impacto operativo. El código se comparte para estudio y revisión metodológica.

## Contenido

- `src/`: pipeline local de datos, variables, modelos, evaluación y ranking.
- `tests/`: pruebas unitarias que usan datos sintéticos o condiciones temporales.
- `config.yaml`: configuración ilustrativa con nombres de archivos genéricos.
- `requirements.txt`: dependencias de Python.

## Datos

**No se publican datos reales, datos procesados, modelos, rankings ni resultados de la planta.** La ejecución completa requiere fuentes privadas con el esquema esperado por el código. Los nombres de archivos en `config.yaml` son ejemplos y deben adaptarse a fuentes autorizadas. No hay un modelo entrenado distribuido con este repositorio.

## Preparación

Desde la raíz del repositorio, con Python 3.11 o superior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

Los comandos del pipeline requieren datos privados preparados bajo `data/raw` y una configuración adecuada. No deben interpretarse como una evaluación reproducible con los archivos públicos.

## Alcance técnico

El procesamiento es local y por lotes. Usa DuckDB y Parquet para las series, modelos de clasificación para estimar riesgo y una política separada para agrupar notificaciones. Los horizontes se etiquetan sobre el intervalo futuro `(t, t+h]`. La disponibilidad de variables se limita a lo que habría estado publicado en el momento de cada predicción.

