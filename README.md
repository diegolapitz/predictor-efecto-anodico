# Predicción de efectos anódicos

Proyecto de investigación para **ordenar cubas de electrólisis por riesgo de iniciar un efecto anódico** en los próximos 15, 30 y 60 minutos. La idea es actualizar el ranking con mediciones registradas aproximadamente cada cinco minutos para ayudar a priorizar la revisión de cubas.

## Qué hice

- Preparé series de mediciones, eventos de efecto anódico y registros operativos para construir ejemplos de entrenamiento sin usar información futura.
- Generé variables de tendencia, variabilidad y estado reciente de cada cuba, y definí etiquetas para los tres horizontes.
- Comparé regresión logística, Extra Trees y LightGBM; LightGBM fue el modelo principal del estudio.
- Implementé evaluación por períodos cronológicos, rankings por cuba y métricas de detección, precisión, anticipación y carga de alarmas.
- Incorporé pruebas de la lógica temporal y del etiquetado.

El análisis abarcó 384 cubas. Se trabajó con entrenamiento de enero a mayo de 2026, validación en junio y evaluación retrospectiva en julio. Parte de los eventos se identificó con una señal indirecta porque el registro oficial no cubría todo el período.

## Qué obtuve y qué funciona hoy

El pipeline **se ejecutó con datos privados** y produjo datasets, modelos, rankings y análisis retrospectivos. En esa exploración, el ranking mostró potencial para priorizar cubas, mientras que la cantidad de avisos de una alarma automática resultó demasiado alta para recomendar su uso operativo.

Una revisión posterior detectó un problema en la disponibilidad temporal de una fuente. El código se corrigió, pero **no se reconstruyó el conjunto completo ni se reentrenaron y evaluaron los modelos**. Por eso, las métricas históricas no representan el rendimiento de la versión actual. No hay una alarma validada ni un servicio conectado a planta.

El código publicado pasa **6 pruebas unitarias** que no necesitan datos reales. La ejecución completa no puede probarse sólo con este repositorio porque las fuentes y los modelos son privados.

## Archivos publicados

- `src/`: preparación de datos, variables, entrenamiento, evaluación y ranking.
- `tests/`: pruebas independientes de los datos de la planta.
- `config.yaml`: configuración ilustrativa con nombres de archivos genéricos.
- `requirements.txt`: dependencias.

**No se incluyen datos reales, modelos entrenados, rankings ni resultados de la planta.** Para ejecutar el pipeline completo se necesitan fuentes autorizadas con el esquema esperado y adaptar `config.yaml`.

## Instalación y pruebas

Con Python 3.11 o superior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```
