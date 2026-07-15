<div align="center">

# Flight Delay Risk

### Workbench bilingüe de riesgo de retraso antes de la salida

Creado por **Oriol Martínez**

Flight Delay Risk prioriza vuelos programados por exposición estimada a retraso, valida cada fila subida, explica el modelo seleccionado y muestra la evidencia temporal del artefacto público.

`English / Español` · `FastAPI` · `Streamlit` · `scikit-learn` · `BTS 2024` · `validación temporal` · `calibración` · `informes PDF` · `monitorización` · `Docker`

![Vista previa de Flight Delay Risk](docs/assets/product_preview.svg)

**[English](README.md) · [Español](README_ES.md)**

</div>

> **Idea central.** La mayoría de demos de retrasos devuelven un score. Flight Delay Risk lo convierte en un flujo de revisión: introduce o sube un horario, prioriza riesgo, inspecciona evidencia, verifica la estabilidad temporal y exporta un informe bilingüe.

## Estado de la versión pública

**Flight Delay Risk v1.5.0 — Self-Explaining Product UI Release** es la versión estable de portfolio. Incluye la familia Extra Trees congelada y reajustada sobre una muestra determinista de 250.000 vuelos, un test final intacto de 50.453 vuelos, política calibrada top-10%, API/UI, contrato OpenAPI, health checks de Docker y evidencia de smoke de producción.

El archivo no incorpora una URL alojada. Está preparado para desplegar, pero el smoke comprometido valida el empaquetado y el contrato runtime, no disponibilidad externa.

La v1.5 incorpora un fondo azul muy claro, superficies blancas, un banner de priorización más compacto, soporte de ruta cualitativo y gráficos temporales comparados con la prevalencia. El artefacto estadístico desplegado sigue siendo el refit escalado de Extra Trees; no se ha vuelto a seleccionar el modelo utilizando el test final.

## Interfaz autoexplicativa

El dashboard está diseñado para entenderse sin leer este README. Cada métrica visible incluye una interpretación de una línea, los acrónimos técnicos se traducen a lenguaje de producto y los diagnósticos brutos permanecen en desplegables **Avanzados**.

## Recorrido del producto

### 1. Analizar un vuelo

El usuario introduce campos naturales del horario. El sistema devuelve:

- probabilidad calibrada de retraso de 15 minutos o más;
- score bruto del modelo;
- tasa histórica y soporte exacto de la ruta;
- exposición relativa frente a la cohorte;
- cobertura de ruta y aerolínea-ruta;
- contribuciones locales firmadas del modelo;
- informe PDF en español o inglés.

Las contribuciones están medidas en log-odds antes de la calibración. Explican el comportamiento del clasificador, no causas reales.

### 2. Priorizar un horario

La plantilla recomendada utiliza:

```csv
flight_number,airline,origin,destination,flight_date,scheduled_departure,scheduled_arrival,scheduled_duration_minutes,distance_miles
418,DL,JFK,LAX,2026-07-18,18:30,21:45,375,2475
```

Flight Delay Risk:

1. normaliza alias soportados;
2. valida cada fila;
3. excluye filas incorrectas sin perder las válidas;
4. detecta rutas no vistas y con poco soporte;
5. realiza inferencia vectorizada;
6. genera probabilidades calibradas y contexto histórico;
7. asigna colas `Priority`, `Watch` y `Routine`;
8. exporta CSV y PDF bilingüe.

### 3. Validación

La interfaz muestra PR-AUC, Lift@10%, Brier, ECE, curva de fiabilidad, tres folds temporales, benchmark de siete modelos y contrato de encoding histórico.

### 4. Modelo y operaciones

La interfaz muestra linaje del artefacto, periodos de datos, predicciones registradas, estado PSI, latencias medidas, model card, API y despliegue.

---

## Resultado honesto

### Refit escalado v1.4

La familia de modelo y la capacidad de revisión del 10% quedaron congeladas antes de esta capa. La muestra crece de 30.000 a **250.000 vuelos**: 168.519 para refit de finalistas, 31.028 para calibración y **50.453** en el test intacto del 19 de octubre al 31 de diciembre.

| Métrica | Artefacto v1.5.0 |
|---|---:|
| ROC-AUC | 0,6179 |
| PR-AUC | 0,2386 |
| Precision@Top10% | 0,2801 |
| Lift@Top10% | 1,639× |
| Brier | 0,1385 |
| ECE | 0,0130 |

La política top-10% revisa 5.046 vuelos, con precisión `0,2800`, recall `0,1639` y lift `1,6384×`. Los intervalos semanales sobre el test ampliado sitúan PR-AUC en `[0,2036, 0,2813]` y Lift@10% en `[1,5096, 1,7510]` con 100 remuestreos.

Extra Trees mantiene la familia, pero usa representación ordinal compacta `float32`; el baseline se denomina explícitamente SGD logistic. Se intentó un build de 500.000 filas, pero el encoder histórico con recencia superó el presupuesto reproducible, por lo que la release se detiene honestamente en 250.000.

Más detalle en [`docs/SCALE_REFIT_AND_DEPLOYMENT.md`](docs/SCALE_REFIT_AND_DEPLOYMENT.md).

### Benchmark público de siete modelos

El zoo público compara un baseline lineal, Random Forest y Extra Trees, XGBoost y LightGBM, una MLP con embeddings y FT-Transformer. Se retiraron Elastic Net, HistGradientBoosting y CatBoost para reducir comparaciones redundantes.

| Modelo | PR-AUC selección | Lift@10% |
|---|---:|---:|
| Extra Trees | 0.3728 | 1.784× |
| Random Forest | 0.3637 | 1.744× |
| Regresión logística | 0.3586 | 1.774× |
| LightGBM | 0.3577 | 1.656× |
| XGBoost | 0.3524 | 1.665× |
| MLP con embeddings | 0.3442 | 1.656× |
| FT-Transformer | 0.3330 | 1.439× |

## Contrato anti-leakage

**Permitido antes de la salida:** aerolínea, ruta, calendario, horas programadas, duración, distancia y agregados históricos construidos con fechas anteriores.

**Bloqueado:** retrasos reales, horas reales, taxi, wheels, tiempo en el aire, causas de retraso, cancelación y desvío como variables de inferencia.

Cada fila de entrenamiento recibe históricos calculados únicamente con `FlightDate` estrictamente anteriores. Las filas del mismo día se transforman juntas.

---

## Arquitectura

![Arquitectura de Flight Delay Risk](docs/assets/architecture.svg)

```text
BTS
  -> fingerprints y detección de meses duplicados
  -> split train / selección / calibración / test
  -> contexto target-free del horario completo
  -> 112 features: calendario + históricos + soporte + recencia + congestión
  -> selección de modelos en bloque separado
  -> selección holdout de calibración y umbral
  -> artefacto versionado
  -> inferencia + explicación local
  -> Streamlit / FastAPI / PDF bilingües
  -> logging + PSI
```

---

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Zoo completo: pip install -r requirements-advanced.txt
streamlit run app/dashboard/streamlit_app.py
uvicorn app.api.main:app --reload
```

Con Docker:

```bash
docker compose up --build
```

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`

## Quality gate

```bash
python -m scripts.quality_gate
```

CI y `make test` ejecutan la suite completa por separado; el gate verifica su evidencia y ejecuta lint, smoke neural con serialización, artefacto, inferencia calibrada, explicaciones locales, PDFs bilingües, reportes temporales y manifest de release.

## Limitaciones

- Sin meteorología en vivo, rotación de aeronave, tripulación, ATC ni estado operacional.
- Artefacto entrenado con BTS; la capa europea es experimental.
- Las contribuciones explican el estimador seleccionado, no mecanismos causales.
- La monitorización es ligera y basada en archivos para una release de portfolio.

Consulta el [README principal](README.md) para la documentación técnica completa.

## Licencia

MIT. Creado por **Oriol Martínez**.
