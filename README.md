# Predicción de efectos anódicos

Código de investigación para estimar el riesgo de inicio de un efecto anódico y ordenar cubas en horizontes de 15, 30 y 60 minutos. El proyecto incluye preparación de series temporales, generación de variables, entrenamiento, evaluación y rankings.

**Estado:** prototipo experimental; no es un sistema de alarmas en producción.

## Contenido

- `src/`: código del pipeline y los modelos.
- `tests/`: pruebas que no requieren datos de la planta.
- `config.yaml`: configuración de ejemplo.
- `requirements.txt`: dependencias.

## Datos y reproducción

Este repositorio no incluye datos reales, modelos entrenados ni resultados de la planta. Para ejecutar el pipeline completo se necesitan fuentes privadas compatibles con el esquema esperado y adaptar `config.yaml`. Por ese motivo, el entrenamiento y sus métricas no se pueden reproducir sólo con los archivos públicos.

## Instalación y pruebas

Con Python 3.11 o superior:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```
