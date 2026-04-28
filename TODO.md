# TODO — Proyecto Final MGE: Producto de Datos 1C Company

> **Deadline:** miércoles 29 de abril 2026 · 23:59 CANVAS  
> **Paulina** → Infraestructura AWS  
> **Andrea** → App Streamlit + Modelo  
> Repo base (tareas previas): `Prediccion_de-_demanda_en_retail/`

---

## Estructura del repo (lo que tiene que existir al entregar)

```
demanda_en_retail_proyecto_final/
│
├── app/                          ← Andrea
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                   (navegación multi-página)
│   └── pages/
│       ├── 1_vp_planeacion.py
│       ├── 2_director_finanzas.py
│       ├── 3_lider_bi.py
│       ├── 4_evaluacion_naive.py
│       └── 5_feedback.py
│
├── infra/                        ← Paulina
│   ├── cloudformation/
│   │   ├── rds.yaml
│   │   └── ecs.yaml
│   └── secrets/
│       └── setup_secrets.sh      (script para crear secret en Secrets Manager)
│
├── etl/                          ← Paulina
│   ├── load_predictions.py       (S3 → RDS: products, predictions, actuals)
│   ├── load_metrics.py           (calcula evaluation_metrics y carga a RDS)
│   └── schema.sql                (CREATE TABLE de las 6 tablas)
│
├── model/                        ← Andrea (si re-entrena)
│   ├── train_improved.py
│   └── evaluate_by_group.py      (RMSE por categoría vs naive)
│
├── diagramas/                    ← Paulina
│   ├── arquitectura.drawio
│   ├── arquitectura.png
│   ├── erd.drawio
│   └── erd.png
│
├── docs/
│   └── reporte.md                ← Ambas
│
├── README.md                     ← Ambas
└── TODO.md                       ← este archivo
```

---

## Qué ya tenemos y dónde está

| Componente | Estado | Ubicación en repo anterior |
|---|---|---|
| Modelo HistGBR serializado | LISTO | `artifacts/models/model.joblib` |
| Predicciones para mes 34 (nov 2015) | LISTO | `data/predictions/predictions.csv` |
| Script de preprocesamiento | LISTO | `src/preprocessing/prep.py` |
| Script de entrenamiento | LISTO | `src/training/train.py` |
| Script de inferencia | LISTO | `src/inference/inference.py` |
| Dockerfiles pipeline ML | LISTO | `src/*/Dockerfile` |
| Container SageMaker BYOC | LISTO | `sagemaker/container/` |
| Notebook SageMaker pipeline | LISTO | `sagemaker/notebook/` |
| Datos raw (kaggle) | LISTO | `data/raw/*.csv` |
| Ground truth mes 33 (oct 2015) | LISTO | dentro de `data/prep/matrix.csv.gz` |
| Streamlit app | **FALTA** | — |
| CloudFormation stacks | **FALTA** | — |
| RDS + schema | **FALTA** | — |
| ETL predictions → RDS | **FALTA** | — |
| Secrets Manager | **FALTA** | — |
| Glue Data Catalog | **FALTA** | — |
| Evaluación RMSE por categoría | **FALTA** | — |
| Diagramas draw.io | **FALTA** | — |
| Reporte + README ejecutivo | **FALTA** | — |
| Video demo | **FALTA** | — |

---

## PAULINA — Infraestructura

### P1 — Diagramas

- [ ] `diagramas/arquitectura.drawio` — diagrama de arquitectura completo
  - Incluir: S3, Glue Catalog, ETL, RDS, Secrets Manager, ECR, ECS Fargate, CloudFormation, usuario
  - Export a PNG → pegar en README
- [ ] `diagramas/erd.drawio` — diagrama entidad-relación
  - 6 tablas: `products`, `predictions`, `actuals`, `evaluation_metrics`, `business_feedback`, `flagged_products`
  - Campos, tipos, PKs, FKs, cardinalidad
  - Export a PNG → pegar en README

---

### P2 — Schema de base de datos

Crear `etl/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS products (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    item_id     VARCHAR(20) NOT NULL,
    category    VARCHAR(100),
    shop_id     VARCHAR(20),
    active      BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS predictions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id       UUID REFERENCES products(id),
    prediction_date  DATE NOT NULL,
    predicted_sales  FLOAT,
    lower_bound      FLOAT,
    upper_bound      FLOAT,
    generated_at     TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS actuals (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id   UUID REFERENCES products(id),
    sale_date    DATE NOT NULL,
    actual_sales FLOAT
);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id   UUID REFERENCES products(id),
    group_key    VARCHAR(50),   -- 'all', 'category:<n>', 'shop:<n>'
    rmse         FLOAT,
    mae          FLOAT,
    naive_rmse   FLOAT,
    evaluated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS business_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id  UUID REFERENCES products(id),
    sentiment   VARCHAR(10) NOT NULL,   -- 'positive' | 'negative'
    observation TEXT,
    created_by  VARCHAR(100),
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS flagged_products (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    product_id  UUID REFERENCES products(id),
    feedback_id UUID REFERENCES business_feedback(id),
    reason      VARCHAR(200),
    resolved    BOOLEAN DEFAULT FALSE,
    flagged_at  TIMESTAMP DEFAULT NOW()
);
```

---

### P3 — CloudFormation

Usar los templates de clase como base. Crear:

- [ ] `infra/cloudformation/rds.yaml`
  - RDS PostgreSQL `db.t3.micro`, single-AZ
  - Security Group que solo acepta el SG del ECS task
  - Parámetro: `DBPassword` (lo pondrá Secrets Manager, no hardcodeado)

- [ ] `infra/cloudformation/ecs.yaml`
  - ECS Cluster + Fargate Service
  - Task Definition: imagen de ECR, 0.5 vCPU / 1 GB RAM
  - IAM Role con permisos: `secretsmanager:GetSecretValue`, `s3:GetObject`
  - Application Load Balancer → URL pública (port 8501 de Streamlit)

Deploy (dos ejecuciones separadas, sin stack maestro):
```bash
aws cloudformation deploy --template-file infra/cloudformation/rds.yaml --stack-name 1c-rds --capabilities CAPABILITY_IAM
aws cloudformation deploy --template-file infra/cloudformation/ecs.yaml --stack-name 1c-ecs --capabilities CAPABILITY_IAM
```

---

### P4 — Secrets Manager

```bash
# infra/secrets/setup_secrets.sh
aws secretsmanager create-secret \
  --name "rds/1c-credentials" \
  --secret-string '{
    "host": "<rds-endpoint>",
    "port": "5432",
    "dbname": "retail_poc",
    "username": "admin",
    "password": "<tu-password>"
  }'
```

---

### P5 — Glue Data Catalog (baja prioridad)

- [ ] Crear database `retail_poc` en Glue
- [ ] Crear tabla `predictions_raw` apuntando al CSV de predicciones en S3
- [ ] Screenshot de la consola para el reporte (es requisito que aparezca en diagrama)

---

### P6 — ETL offline (crítico)

`etl/load_predictions.py`:

```python
# Flujo:
# 1. Leer model.joblib desde S3
# 2. Leer data/raw/*.csv desde S3
# 3. Generar predicciones para mes 34 (nov 2015) con intervalos de confianza
# 4. Insertar en RDS: products → predictions → actuals (mes 33) → evaluation_metrics

# Conexión a RDS usando Secrets Manager:
import boto3, json, psycopg2

def get_conn():
    secret = json.loads(
        boto3.client("secretsmanager").get_secret_value(
            SecretId="rds/1c-credentials"
        )["SecretString"]
    )
    return psycopg2.connect(
        host=secret["host"], port=secret["port"],
        dbname=secret["dbname"], user=secret["username"],
        password=secret["password"]
    )
```

- [ ] Cargar `products` (item_id, category via item_categories.csv, shop_id)
- [ ] Cargar `predictions` (mes 34 = noviembre 2015, con lower/upper bound)
- [ ] Cargar `actuals` (mes 33 = octubre 2015, del ground truth)
- [ ] Cargar `evaluation_metrics` (RMSE por categoría vs naive — Andrea genera esto)

---

### P7 — Evidencias para el reporte

Screenshots de consola AWS que hay que tomar:
- [ ] ECS → servicio corriendo (status "RUNNING")
- [ ] CloudFormation → stack `CREATE_COMPLETE`
- [ ] ECR → imagen `1c-app:latest` publicada
- [ ] RDS → instancia "available"
- [ ] URL pública de la app abierta en browser

---

## ANDREA — App Streamlit + Modelo

### A1 — Evaluación del modelo por grupo (urgente)

Crear `model/evaluate_by_group.py`:

```python
# 1. Cargar matrix.csv.gz (tiene ground truth del mes 33)
# 2. Generar predicciones del modelo para mes 33
# 3. Calcular naive forecast (lag_1 = valor del mes 32)
# 4. Calcular RMSE por item_category_id y por shop_id
# 5. Guardar CSV: category_id, rmse_model, rmse_naive, mejora_%
# 6. Si alguna categoría tiene rmse_model > rmse_naive → re-entrenar ese grupo
```

**Entregable:** tabla `evaluation_by_category.csv` que Paulina carga a `evaluation_metrics` en RDS.

Si el modelo no supera naive en categorías relevantes → re-entrenar con features adicionales (ej. media móvil por categoría, tendencia).

---

### A2 — Estructura base del Streamlit

```bash
app/
├── Dockerfile
├── requirements.txt          # streamlit, psycopg2-binary, boto3, pandas, plotly
├── main.py
└── pages/
    ├── 1_vp_planeacion.py
    ├── 2_director_finanzas.py
    ├── 3_lider_bi.py
    ├── 4_evaluacion_naive.py
    └── 5_feedback.py
```

`app/main.py`:
```python
import streamlit as st
st.set_page_config(page_title="1C Company — POC Pronósticos", layout="wide")
st.title("Producto de Datos — 1C Company")
st.info("Selecciona una vista del menú lateral.")
```

Empieza con datos mock (CSV hardcodeado) — conectar a RDS cuando Paulina lo tenga listo.

---

### A3 — Vistas del Streamlit

#### Vista 1 — VP de Planeación (`1_vp_planeacion.py`)

**Wireframe:**
```
┌──────────────────────────────────────────────────────┐
│  VP de Planeación — Pronóstico Mensual               │
├──────────────────────────────────────────────────────┤
│  Tienda [dropdown] Categoría [dropdown]              │
│  Fecha [date]      Temporada [dropdown: Q1/Q2/Q3/Q4] │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Gráfica de líneas: histórico (12 meses) +     │  │
│  │  pronóstico mes siguiente con banda confianza  │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  [Export]  →  popup: ✓ Descarga completa            │
└──────────────────────────────────────────────────────┘
```

**Conexión a RDS:**
```python
# Consulta principal
SELECT p.predicted_sales, p.lower_bound, p.upper_bound,
       a.actual_sales, a.sale_date
FROM predictions p
JOIN products pr ON p.product_id = pr.id
LEFT JOIN actuals a ON a.product_id = pr.id
WHERE pr.shop_id = %s AND pr.category = %s
ORDER BY a.sale_date

# Temporada → mapea a rango de meses del histórico (actuals)
# Q1=Ene-Mar, Q2=Abr-Jun, Q3=Jul-Sep, Q4=Oct-Dic
```

**Tablas usadas:** `predictions` (read), `actuals` (read), `products` (join)

---

#### Vista 2 — Director de Finanzas (`2_director_finanzas.py`)

**Wireframe:**
```
┌──────────────────────────────────────────────────────┐
│  Director de Finanzas — Export Batch                 │
├──────────────────────────────────────────────────────┤
│  Categoría [dropdown]   Tienda [dropdown]            │
│  [Seleccionar todos]                                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ item_id | categoría | tienda | predicción    │   │
│  │ ...     │ ...       │ ...   │ ...            │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  [Descargar .csv]    [Descargando... ████████ 100%] │
└──────────────────────────────────────────────────────┘
```

**Conexión a RDS:**
```python
# Consulta batch (puede ser miles de filas)
SELECT pr.item_id, pr.category, pr.shop_id,
       p.predicted_sales, p.lower_bound, p.upper_bound
FROM predictions p
JOIN products pr ON p.product_id = pr.id
WHERE pr.category = %s   -- o sin filtro si "seleccionar todos"
ORDER BY pr.category, pr.shop_id

# st.download_button genera el CSV en memoria (no guarda en servidor)
```

**Tablas usadas:** `predictions` (read), `products` (join)

---

#### Vista 3 — Líder BI (`3_lider_bi.py`)

**Wireframe:**
```
┌──────────────────────────────────────────────────────┐
│  Exploración de Pronósticos                          │
├──────────────────────────────────────────────────────┤
│  Categoría [ ]   Tienda [ ]   Rango ventas [ — ]    │
│                                                      │
│  item_id | category | shop | pred | lower | upper   │
│  ──────────────────────────────────────────────────  │
│  ...      │ ...      │ ...  │ ...  │ ...   │ ...    │
│                                                      │
│  [Exportar a CSV]                                    │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Dashboard: gráficas agregadas por categoría   │  │
│  │  (bar chart ventas predichas por categoría)    │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**Conexión a RDS:**
```python
# Tabla filtrable
SELECT pr.item_id, pr.category, pr.shop_id,
       p.predicted_sales, p.lower_bound, p.upper_bound
FROM predictions p
JOIN products pr ON p.product_id = pr.id

# Filtros aplicados en pandas después del query (o con WHERE dinámico)
# Dashboard: GROUP BY pr.category → SUM(p.predicted_sales)
```

**Tablas usadas:** `predictions` (read), `products` (join)

---

#### Vista 4 — Evaluación vs Naive (`4_evaluacion_naive.py`)

**Wireframe:**
```
┌──────────────────────────────────────────────────────┐
│  Evaluación del Modelo                               │
├──────────────────────────────────────────────────────┤
│  RMSE modelo [  2.94  ]   Costo de ahorro [  X%  ]  │
│  RMSE naive  [  6.29  ]                              │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │  Scatter: predicción vs real                   │  │
│  │  (cada punto = un par producto-tienda)         │  │
│  │  Diagonal perfecta en rojo                     │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  Tabla: category | RMSE modelo | RMSE naive | mejora│
└──────────────────────────────────────────────────────┘
```

**Conexión a RDS:**
```python
# KPIs globales
SELECT group_key, rmse, naive_rmse, mae
FROM evaluation_metrics
WHERE group_key = 'all'

# Tabla por categoría
SELECT group_key, rmse, naive_rmse,
       ROUND((1 - rmse/naive_rmse)*100, 1) AS mejora_pct
FROM evaluation_metrics
WHERE group_key LIKE 'category:%'
ORDER BY rmse DESC

# Scatterplot: predicciones vs actuals del mes 33
SELECT p.predicted_sales, a.actual_sales, pr.category
FROM predictions p
JOIN products pr ON p.product_id = pr.id
JOIN actuals a ON a.product_id = pr.id
```

**Tablas usadas:** `evaluation_metrics` (read), `predictions` (read), `actuals` (read), `products` (join)

---

#### Vista 5 — Feedback + Alertas (`5_feedback.py`)

**Wireframe:**
```
┌──────────────────────────────────────────────────────┐
│  Feedback de Producto                                │
├──────────────────────────────────────────────────────┤
│  ID Producto [dropdown]                              │
│  Tipo de producto [dropdown de categoría]            │
│  Feedback:                                           │
│  ┌──────────────────────────────────────────┐        │
│  │ [text area]                              │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  [Feedback positivo]    [Feedback negativo]          │
│                                                      │
│  ───────────────────────────────────────────         │
│  Productos con alerta                                │
│  ID producto │ Alerta                                │
│  ─────────────────────                               │
│  ...         │ ...                                   │
└──────────────────────────────────────────────────────┘
```

**Conexión a RDS:**
```python
# Guardar feedback (botón positivo/negativo)
INSERT INTO business_feedback (product_id, sentiment, observation, created_by)
VALUES (%s, %s, %s, %s)

# Si sentiment = 'negative' → también insertar en flagged_products
INSERT INTO flagged_products (product_id, feedback_id, reason)
VALUES (%s, %s, %s)

# Tabla "Productos con alerta" (read)
SELECT pr.item_id, fp.reason, fp.flagged_at, fp.resolved
FROM flagged_products fp
JOIN products pr ON fp.product_id = pr.id
WHERE fp.resolved = FALSE
ORDER BY fp.flagged_at DESC
```

**Tablas usadas:** `business_feedback` (write), `flagged_products` (write + read), `products` (join)

---

### A4 — Dockerfile de la app

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "main.py",
            "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build y push a ECR
docker build -t 1c-app:latest ./app/
docker tag 1c-app:latest <account>.dkr.ecr.us-east-1.amazonaws.com/1c-app:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <ecr-url>
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/1c-app:latest
```

---

### A5 — Conexión a RDS desde Streamlit (helper compartido)

Crear `app/db.py`:

```python
import boto3, json, psycopg2, streamlit as st

@st.cache_resource
def get_connection():
    """Obtiene conexión a RDS usando credenciales de Secrets Manager."""
    client = boto3.client("secretsmanager", region_name="us-east-1")
    secret = json.loads(
        client.get_secret_value(SecretId="rds/1c-credentials")["SecretString"]
    )
    return psycopg2.connect(
        host=secret["host"], port=int(secret["port"]),
        dbname=secret["dbname"], user=secret["username"],
        password=secret["password"], connect_timeout=5
    )

def query(sql: str, params=None) -> list[dict]:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

def execute(sql: str, params=None):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()
```

**Mientras no hay RDS lista** → usar mock:
```python
# En cada página, al inicio:
import os
USE_MOCK = os.getenv("USE_MOCK", "true") == "true"
```

---

## Arquitectura de referencia

```
                         ┌─────────────────────────────┐
  Datos fuente           │  INFRAESTRUCTURA (Paulina)  │
  (repo anterior)        └─────────────────────────────┘
       │
       ▼
  ┌─────────┐    ┌──────────────┐
  │   S3    │◄───│ Glue Catalog │  (catálogo, apunta a S3)
  │ modelos │    └──────────────┘
  │  CSVs   │
  └────┬────┘
       │
       ▼
  ┌──────────┐
  │ ETL      │  (load_predictions.py)
  │ offline  │  Lee modelo de S3, genera predicciones,
  │ (script) │  carga todo a RDS
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │   RDS    │  PostgreSQL — 6 tablas
  │PostgreSQL│  products / predictions / actuals /
  └────┬─────┘  evaluation_metrics / business_feedback / flagged_products
       │
       ▼
  ┌────────────────────────────────────────┐
  │           ECS FARGATE                  │
  │  ┌──────────────┐  ┌────────────────┐  │
  │  │ Secrets Mgr  │  │  Streamlit App │  │
  │  │ (creds RDS)  │→ │   5 vistas     │  │
  │  └──────────────┘  └────────────────┘  │
  └──────────┬─────────────────────────────┘
             │  imagen de
        ┌────┴──────┐      ┌──────────────────┐
        │    ECR    │      │  CloudFormation   │
        │  imagen   │      │  (define todo)    │
        └───────────┘      └──────────────────┘
             │
             ▼
      URL pública → usuarios de negocio
```

---

## Prioridades si el tiempo aprieta

| Si solo queda tiempo para... | Recortar esto |
|---|---|
| 1 día de trabajo | Quitar Dashboard de Vista 3, dejar solo tabla |
| Medio día | Combinar vistas 2 y 3 en una sola |
| Pocas horas | Dejar solo vistas 1, 4 y 5 (las más evaluadas) |
| Emergencia | Vistas con datos mock pero app en Fargate funcionando |

**Lo que NUNCA se puede recortar:**
- App corriendo en Fargate con URL pública
- Vista 4 (evaluación vs naive) con datos reales
- Sección "Uso de IA en el proyecto" en el reporte

---

## Entregables para CANVAS

```
□ URL del repo en GitHub (público o con acceso al profe)
□ URL pública de la app en ECS Fargate (debe seguir viva hasta el 1 de mayo)
□ README ejecutivo con diagramas embebidos y screenshots de AWS
□ arquitectura.drawio + .png
□ erd.drawio + .png
□ reporte.md
□ Video demo de las 5 vistas con datos reales
```

El reporte debe incluir: problema de negocio · arquitectura · modelo de datos · pipeline ML · evaluación vs naive · tour de la app · screenshots AWS · costo mensual · limitaciones · **sección "Uso de IA en el proyecto"** (obligatoria).

---

*Última actualización: 2026-04-25*
