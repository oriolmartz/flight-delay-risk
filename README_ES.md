<div align="center">

[Fuente BTS](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ) · **7.079.081 filas fuente** · **refit desplegado con 168.519 vuelos** · **test intacto de 50.453 vuelos**

# Flight Delay Risk

### Workbench de riesgo de retraso antes de la salida

**Apoyo a decisiones antes de la salida para operaciones aéreas con capacidad limitada.**

Encuentra qué vuelos programados merecen atención primero, entiende por qué se han priorizado y revisa la evidencia que sostiene la release.

[English](README.md) · [Español](README_ES.md) · [Guía de datos](docs/DATA.md) · [Model card](docs/MODEL_CARD.md) · [Contrato API](docs/openapi.json)

![Landing de Flight Delay Risk](docs/assets/readme_landing.png)

`Python` · `scikit-learn` · `PyTorch` · `FastAPI` · `Streamlit` · `Docker`

</div>

## Empieza aquí

Un equipo de operaciones no puede investigar todas las salidas con el mismo nivel de atención. Flight Delay Risk convierte un horario publicado en una cola de revisión antes del despegue:

```text
vuelo programado → riesgo calibrado → evidencia histórica → cola de revisión
```

El producto responde una pregunta práctica:

> **¿Qué vuelos programados debería revisar primero un analista cuando la capacidad es limitada?**

| Contrato de producto | Qué significa |
|---|---|
| **Usuario** | Analista de operaciones, control de red o gestión de disrupciones. |
| **Target** | Probabilidad de llegar con al menos 15 minutos de retraso (`ArrDel15`). |
| **Inputs** | Aerolínea, ruta, fecha, horarios programados, duración y distancia. |
| **Acción** | Priorizar el 10% de mayor riesgo dentro del horario cargado. |
| **Evidencia** | Baseline de ruta, soporte histórico, contribuciones locales y validación temporal. |
| **Límite** | Sin meteorología en vivo, rotación de aeronaves, tripulación, ATC ni datos posteriores a la salida. |

Es un **workbench de triaje**, no una promesa de que un vuelo se retrasará ni un sistema de seguridad o despacho.

## Resultado honesto

El artefacto Extra Trees desplegado se reajustó con **168.519 vuelos**, se calibró con **31.028 vuelos posteriores** y se evaluó una sola vez en un test intacto de **50.453 vuelos** entre el 19 de octubre y el 31 de diciembre de 2024.

| Señal del test final | Resultado | Lectura operativa |
|---|---:|---|
| **Lift en el top 10%** | **1,64×** | La cola contiene alrededor de un 64% más de vuelos retrasados que una selección aleatoria. |
| **Precisión en el top 10%** | **28,0%** | Aproximadamente 28 de cada 100 vuelos priorizados sufrieron retraso. |
| **PR-AUC** | **0,239** | Mejor ranking que la prevalencia del test, **17,1%**, y que el baseline logístico, **0,203**. |
| **ROC-AUC** | **0,618** | Discriminación moderada utilizando solo información programada. |
| **Error de calibración (ECE)** | **0,013** | Las probabilidades siguieron de cerca las frecuencias observadas en el test final. |

El resultado es útil para **ordenar atención limitada**, no para afirmar con seguridad qué ocurrirá con un vuelo concreto. El rendimiento cambia con el tiempo y la auditoría versionada señala actualmente drift alto. Esa limitación forma parte de la evidencia del producto.

## Tour del producto

### 1. Ver la decisión antes que la maquinaria

La landing explica primero la pregunta operativa, el alcance y el estado de la release. Un ejemplo real distingue entre probabilidad absoluta, tasa habitual de la ruta y soporte histórico disponible.

### 2. Analizar un vuelo

Introduce campos naturales del horario; las features de calendario y del modelo se derivan automáticamente. El resultado devuelve:

- probabilidad calibrada de llegar con 15 o más minutos de retraso;
- estado `Prioridad`, `Vigilancia` o `Rutina`;
- riesgo frente al baseline histórico de la ruta;
- número de vuelos previos que sustentan esa referencia;
- factores que elevaron y redujeron la estimación;
- informe PDF bilingüe.

![Resumen de decisión para un vuelo con evidencia histórica](docs/assets/readme_analyze.png)

### 3. Explorar el histórico aeroportuario sin salir del flujo

Haz scroll bajo el formulario para explorar la evidencia de entrenamiento por origen o destino. El color representa la proporción histórica de vuelos retrasados y el tamaño, el soporte. Al pasar el ratón aparecen el código, la tasa y el número de vuelos históricos.

![Mapa de calor histórico de aeropuertos](docs/assets/readme_heatmap.png)

El mapa describe **evidencia histórica de entrenamiento de BTS**. No representa meteorología en vivo, congestión actual ni una predicción por sí solo.

### 4. Priorizar un horario completo

Sube la plantilla CSV o un horario válido. Se conservan las filas correctas aunque otras fallen. El workbench marca rutas con poco soporte, ordena el riesgo calibrado, aplica el presupuesto de revisión del 10% y exporta CSV e informes PDF bilingües.

![Horario priorizado bajo una capacidad de revisión limitada](docs/assets/readme_rank.png)

### 5. Inspeccionar validación y operaciones

Las dos últimas vistas exponen folds cronológicos, calibración, comparación con baselines, lineage del modelo, endpoints, salud de release y evidencia de despliegue. La historia técnica está disponible sin bloquear el flujo principal.

![Métricas del test final intacto y guía de interpretación](docs/assets/readme_validation.png)

## Flujo de producto

El score del modelo y la decisión de negocio se mantienen separados:

```text
1. Estimar      ¿Qué probabilidad hay de un retraso de 15+ minutos?
2. Contexto     ¿Es un riesgo inusual para esta ruta?
3. Soporte      ¿Cuánta evidencia histórica permitida existe?
4. Restringir   ¿Cuántos vuelos puede revisar realmente el equipo?
5. Priorizar    ¿Qué vuelos entran en la cola?
6. Monitorizar  ¿Está empeorando la calibración o aparece drift?
```

Así, el mismo modelo calibrado puede soportar otra capacidad o estructura de costes sin fingir que el clasificador conoce la decisión operativa.

## Cómo funciona el sistema

![Arquitectura de Flight Delay Risk](docs/assets/architecture.svg)

```text
registros mensuales BTS
→ validación, limpieza y fingerprinting
→ bloques cronológicos de train / selección / calibración / test
→ features de horario, histórico, recencia y congestión
→ comparación de familias y refit escalado
→ calibración sigmoid
→ política top-k de revisión
→ FastAPI / Streamlit / PDF
→ health checks, logging y monitorización de drift
```

## Datos y lineage de la release

La fuente y la muestra utilizada por el artefacto público están relacionadas, pero no son la misma cifra.

| Capa | Filas | Función |
|---|---:|---|
| **Fuente BTS** | **7.079.081** | Doce archivos mensuales de Reporting Carrier On-Time Performance de 2024. |
| **Dataset canónico limpio** | **6.965.267** | Registros supervisados válidos con cobertura de los 366 días de 2024. |
| **Muestra de release** | **250.000** | Build determinista con el protocolo cronológico congelado. |
| **Refit desplegado** | **168.519** | Entrenamiento más bloque de selección incorporado al refit final. |
| **Calibración** | **31.028** | Holdout posterior para elegir sigmoid y reajustarlo. |
| **Test final** | **50.453** | Ventana intacta entre octubre y diciembre. |

El target es `ArrDel15 = 1` cuando la llegada tiene al menos 15 minutos de retraso. Los CSV crudos y el parquet se excluyen deliberadamente de Git; el manifiesto registra hashes, filas, esquema, cobertura y fingerprint del dataset procesado.

- [Descargar registros BTS](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ)
- [Leer el contrato de datos](docs/DATA.md)
- [Inspeccionar el manifiesto procesado](data/processed/data_manifest.json)

## Comparación de modelos

El model zoo compara paradigmas reconocibles bajo el mismo protocolo cronológico.

| Paradigma | Modelos |
|---|---|
| Baseline interpretable | Logistic Regression |
| Bagging | Random Forest, Extra Trees |
| Gradient boosting | XGBoost, LightGBM |
| Neural tabular | MLP con embeddings, FT-Transformer |

Extra Trees ganó la regla declarada y quedó congelado antes del refit escalado. En tres folds temporales posteriores, MLP, FT-Transformer y Extra Trees ganaron una vez cada uno: ninguna familia dominó todos los periodos.

![Comparación cronológica de selección entre siete modelos](docs/assets/readme_model_comparison.png)

La captura corresponde al **bloque cronológico de selección** utilizado para elegir el modelo. Se mantiene deliberadamente separado de los resultados del test final intacto que aparecen al principio del README.

<details>
<summary><strong>Benchmark de selección</strong></summary>

| Modelo | PR-AUC | Lift@10% |
|---|---:|---:|
| **Extra Trees** | **0,3728** | **1,784×** |
| Random Forest | 0,3637 | 1,744× |
| Logistic Regression | 0,3586 | 1,774× |
| LightGBM | 0,3577 | 1,656× |
| XGBoost | 0,3524 | 1,665× |
| MLP con embeddings | 0,3442 | 1,656× |
| FT-Transformer | 0,3330 | 1,439× |

Son métricas del **bloque de selección**, no afirmaciones sobre el test final. El resultado posterior se muestra al principio del README.

</details>

## Diseño de validación

La release impone evaluación únicamente hacia delante:

```text
entrenamiento              2024-01-01 → 2024-07-16
selección / refit final    2024-07-17 → 2024-09-04
calibración                2024-09-05 → 2024-10-18
test final intacto         2024-10-19 → 2024-12-31
```

- El modelo se elige antes del test final.
- Los candidatos de calibración se comparan en un holdout posterior; gana **sigmoid** y se reajusta en el bloque permitido.
- La política operativa se congela antes del informe final.
- Los intervalos de confianza usan 100 muestras de block bootstrap semanal.
- La release incluye tres folds temporales, robustez, ablaciones y estabilidad de features.

## Contrato anti-leakage

Solo puede influir en la predicción información disponible antes de la salida.

- Las features históricas derivadas del target usan **fechas `FlightDate` estrictamente anteriores**.
- Una etiqueta nunca construye features para otro vuelo del mismo día.
- Validación, calibración y test usan mapas ajustados solo con periodos anteriores permitidos.
- Aerolíneas, aeropuertos y rutas desconocidos reciben fallbacks suavizados explícitos.
- Se bloquean retrasos reales, horas reales, taxi, wheels, cancelación, desvío y causas de retraso.

La explicación local es una descomposición de caminos de árboles reescalada a log-odds. Explica el comportamiento del modelo, no mecanismos causales.

## Superficies de producto y API

| Superficie | Función |
|---|---|
| **Streamlit** | Análisis bilingüe, ranking, heatmap, validación y evidencia de release. |
| **FastAPI** | Contratos tipados de predicción, ranking, informes, metadata, salud y monitorización. |
| **PDF / CSV** | Briefs portables y horarios priorizados en inglés y español. |
| **Operaciones** | `/live`, `/ready`, request IDs, latencia, logging y PSI. |

<details>
<summary><strong>Endpoints públicos</strong></summary>

```text
GET  /live
GET  /ready
GET  /model/info
GET  /model/card
POST /predict
POST /predict/batch
POST /rank
POST /reports/flight
POST /reports/schedule
GET  /monitoring/summary
GET  /monitoring/drift
```

El contrato OpenAPI exportado está en [`docs/openapi.json`](docs/openapi.json).

</details>

## Ejecutar localmente

El artefacto entrenado está incluido; no hace falta reentrenar para utilizar el producto.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Arranca la API:

```bash
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

En otra terminal, arranca el dashboard:

```bash
python -m streamlit run app/dashboard/streamlit_app.py
```

Abre `http://localhost:8501` para el dashboard y `http://localhost:8000/docs` para el contrato API. También puedes ejecutar ambos servicios con:

```bash
docker compose up --build
```

## Evidencia de ingeniería

La release pública incluye **108 tests superados** y evidencia versionada detrás del artefacto.

<details>
<summary><strong>Evaluación, robustez y release</strong></summary>

- [`reports/metrics.json`](reports/metrics.json)
- [`reports/candidate_benchmark.md`](reports/candidate_benchmark.md)
- [`reports/temporal_backtest.md`](reports/temporal_backtest.md)
- [`reports/calibration_report.md`](reports/calibration_report.md)
- [`reports/feature_ablation.md`](reports/feature_ablation.md)
- [`reports/feature_stability.md`](reports/feature_stability.md)
- [`reports/operational_policy.md`](reports/operational_policy.md)
- [`reports/robustness_audit.md`](reports/robustness_audit.md)
- [`reports/drift_analysis.md`](reports/drift_analysis.md)
- [`reports/production_smoke.json`](reports/production_smoke.json)
- [`RELEASE_MANIFEST.json`](RELEASE_MANIFEST.json)

</details>

## Estructura del repositorio

```text
app/api/           contratos públicos y transporte FastAPI
app/dashboard/     interfaz bilingüe de decisión
app/services/      predicción e informes
src/data/          ingestión, limpieza, manifiestos y splits temporales
src/features/      horario, histórico, recencia y congestión
src/models/        entrenamiento, calibración, política y explicaciones
src/monitoring/    logs, robustez y drift
scripts/           workflows reproducibles de entrenamiento y release
reports/           evidencia versionada del artefacto público
docs/              model card, guía de datos, despliegue y limitaciones
```

## Limitaciones

- Los inputs schedule-only no observan meteorología, rotación, tripulación, ATC ni disrupciones activas.
- El ranking cambia con el tiempo; ninguna familia dominó todos los folds.
- La evidencia histórica puede ser débil para combinaciones raras o desconocidas.
- El drift actual es alto; un uso operativo exigiría retraining y revisión de umbral.
- Las contribuciones locales describen al modelo, no por qué ocurren los retrasos.
- El repositorio está preparado para despliegue, pero no declara una URL pública sin verificar uptime.

## Qué demuestra este proyecto

- ML engineering end-to-end sobre registros públicos reales;
- separación de predicción, evidencia, política y acción;
- validación temporal y prevención de leakage;
- comparación de modelos clásicos, boosting y redes tabulares;
- calibración, explicaciones, incertidumbre y análisis de drift;
- ranking operativo bajo restricción de capacidad;
- API, dashboard bilingüe, PDF, Docker, CI y evidencia de release;
- comunicación honesta de rendimiento moderado y limitaciones.

## Licencia

MIT. Desarrollado por **Oriol Martínez**.
