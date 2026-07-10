<div align="center">

# FlightRisk

### Workbench bilingüe de riesgo de retraso antes de la salida

Creado por **Oriol Martínez**

FlightRisk prioriza vuelos programados por exposición estimada a retraso, valida cada fila subida, explica el modelo lineal seleccionado y muestra la evidencia temporal del artefacto público.

`English / Español` · `FastAPI` · `Streamlit` · `scikit-learn` · `BTS 2024` · `validación temporal` · `calibración` · `informes PDF` · `monitorización` · `Docker`

![Vista previa de FlightRisk](docs/assets/product_preview.svg)

**[English](README.md) · [Español](README_ES.md)**

</div>

> **Idea central.** La mayoría de demos de retrasos devuelven un score. FlightRisk lo convierte en un flujo de revisión: introduce o sube un horario, prioriza riesgo, inspecciona evidencia, verifica la estabilidad temporal y exporta un informe bilingüe.

## Estado de la versión pública

FlightRisk v1.0.0 es la versión estable de portfolio. El archivo incluye el artefacto entrenado, evidencia de validación, interfaz bilingüe, exportación PDF, API, monitorización y configuración de despliegue.

No se incluye una URL alojada ficticia. Tras desplegarlo, añade los enlaces públicos del dashboard y la API aquí y en `docs/PUBLIC_RELEASE.md`.

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

FlightRisk:

1. normaliza alias soportados;
2. valida cada fila;
3. excluye filas incorrectas sin perder las válidas;
4. detecta rutas no vistas y con poco soporte;
5. realiza inferencia vectorizada;
6. genera probabilidades calibradas y contexto histórico;
7. asigna colas `Priority`, `Watch` y `Routine`;
8. exporta CSV y PDF bilingüe.

### 3. Validación

La interfaz muestra PR-AUC, Lift@10%, Brier, ECE, curva de fiabilidad, cuatro folds temporales, benchmark de modelos y contrato de encoding histórico.

### 4. Modelo y operaciones

La interfaz muestra linaje del artefacto, periodos de datos, predicciones registradas, estado PSI, latencias medidas, model card, API y despliegue.

---

## Resultado honesto

| Métrica holdout | Resultado |
|---|---:|
| ROC-AUC | 0.6023 |
| PR-AUC | 0.2124 |
| Precision@Top10% | 0.2505 |
| Lift@Top10% | 1.557× |
| Brier score | 0.1336 |
| ECE | 0.0229 |

El rendimiento discriminativo es útil pero modesto. FlightRisk no se presenta como un sistema de predicción de retrasos resuelto. Su valor está en el protocolo temporal, la calibración, el contexto auditable y la entrega con forma de producto.

### Impacto de la calibración

| Métrica | Score bruto | Probabilidad calibrada |
|---|---:|---:|
| Brier | 0.3036 | **0.1336** |
| ECE | 0.3947 | **0.0229** |
| Log loss | 0.8178 | **0.4378** |

### Estabilidad temporal

```text
L1 Logistic Regression seleccionado: 4 / 4 folds
Calibración isotónica seleccionada:   4 / 4 folds
```

Extra Trees ganó por poco un bloque aislado de validación. L1 Logistic se despliega porque fue estable en los cuatro folds temporales y permite explicaciones locales directas.

---

## Contrato anti-leakage

**Permitido antes de la salida:** aerolínea, ruta, calendario, horas programadas, duración, distancia y agregados históricos construidos con fechas anteriores.

**Bloqueado:** retrasos reales, horas reales, taxi, wheels, tiempo en el aire, causas de retraso, cancelación y desvío como variables de inferencia.

Cada fila de entrenamiento recibe históricos calculados únicamente con `FlightDate` estrictamente anteriores. Las filas del mismo día se transforman juntas.

---

## Arquitectura

![Arquitectura de FlightRisk](docs/assets/architecture.svg)

```text
BTS
  -> limpieza y eliminación de leakage
  -> split temporal por fechas completas
  -> encoding histórico con fechas anteriores
  -> selección de modelos
  -> calibración en validación
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

Verifica tests, lint, artefacto, inferencia calibrada, explicaciones locales, PDFs bilingües, reportes temporales y manifest de release.

## Limitaciones

- Sin meteorología en vivo, rotación de aeronave, tripulación, ATC ni estado operacional.
- Artefacto entrenado con BTS; la capa europea es experimental.
- Las contribuciones explican el clasificador, no mecanismos causales.
- La monitorización es ligera y basada en archivos para una release de portfolio.

Consulta el [README principal](README.md) para la documentación técnica completa.

## Licencia

MIT. Creado por **Oriol Martínez**.
