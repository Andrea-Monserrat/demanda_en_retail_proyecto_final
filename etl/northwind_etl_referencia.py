# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Demo: ETL Northwind — PostgreSQL → S3 → Athena
#
# ## Arquitectura: Bronze → Silver → Gold (Medallion Architecture)

# %%
from IPython.display import HTML

HTML("""
<pre class="mermaid" style="background:white;">
%%{init: {'theme':'base', 'themeVariables': {
    'primaryColor':'#e3f2fd','primaryTextColor':'#000',
    'primaryBorderColor':'#1565c0','lineColor':'#546e7a'
}}}%%
graph TB
    PG[(PostgreSQL\nAmazon RDS)]
    ETL[ETL\nSageMaker Studio\nSQLAlchemy + awswrangler]

    subgraph S3["Amazon S3 — Data Lake"]
        direction TB
        BRONZE["🥉 Bronze\nDatos crudos\nde la fuente"]
        SILVER["🥈 Silver\nLimpios y\nvalidados"]
        GOLD["🥇 Gold\nListos para\nanalítica"]
        BRONZE --> SILVER --> GOLD
    end

    GLUE[AWS Glue\nData Catalog]
    ATHENA[Amazon Athena\nSQL Analytics]

    PG --> ETL --> BRONZE
    GOLD --> GLUE --> ATHENA

    classDef postgres fill:#f3e5f5,stroke:#7b1fa2,color:#000
    classDef etl fill:#e8eaf6,stroke:#3949ab,color:#000
    classDef bronze fill:#fff3e0,stroke:#e65100,color:#000
    classDef silver fill:#eceff1,stroke:#546e7a,color:#000
    classDef gold fill:#fff8e1,stroke:#f9a825,color:#000
    classDef glue fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef athena fill:#e3f2fd,stroke:#1565c0,color:#000

    class PG postgres
    class ETL etl
    class BRONZE bronze
    class SILVER silver
    class GOLD gold
    class GLUE glue
    class ATHENA athena
</pre>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: false });
  await mermaid.run();
</script>
""")

# ## Herramientas
#
# | Herramienta        | Rol                                              |
# |--------------------|--------------------------------------------------|
# | `SQLAlchemy 2.0`   | ORM para PostgreSQL: schema + CRUD               |
# | `psycopg2`         | Driver PostgreSQL para Python                    |
# | `pandas`           | Extract con `read_sql`, Transform con DataFrames |
# | `awswrangler`      | Load a S3/Glue, queries a Athena                 |

# %% [markdown]
# ## Instalación de dependencias
#
# Ejecutar esta celda una sola vez. `%pip` instala en el kernel activo de SageMaker Studio.
# No usar `!pip` — puede instalar en el Python incorrecto.

# %%
# !pip install -r requirements.txt -q

# %%
# !unzip northwind_clean.zip


# %% [markdown]
# ## Parte 1: PostgreSQL con Amazon RDS
#
# ### 1.1 Conexión con SQLAlchemy

# %%
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# OJO: en producción usar variables de entorno, no credenciales en texto plano
RDS_ENDPOINT = "northwind.xxxx.us-east-1.rds.amazonaws.com"  # el instructor provee este valor
DB_NAME      = "northwind"
DB_USER      = "itam"
DB_PASSWORD  = "itam2026"
DB_PORT      = 5432

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{RDS_ENDPOINT}:{DB_PORT}/{DB_NAME}"
)

# Verificar conexión
with engine.connect() as conn:
    print("✓ Conexión exitosa a RDS PostgreSQL")

# %% [markdown]
# ### 1.2 Definir Schema con SQLAlchemy 2.0 (Declarative Base)
#
# Modelamos las 5 tablas de Northwind como clases Python usando la API moderna
# de SQLAlchemy 2.0: `Mapped` + `mapped_column`.

# %%
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import String, Integer, Float, Date, ForeignKey
from typing import Optional

class Base(DeclarativeBase):
    pass

class Customer(Base):
    __tablename__ = "customers"
    customerid:   Mapped[str]            = mapped_column(String(5),   primary_key=True)
    companyname:  Mapped[str]            = mapped_column(String(100))
    contactname:  Mapped[Optional[str]]  = mapped_column(String(100))
    country:      Mapped[Optional[str]]  = mapped_column(String(50))
    city:         Mapped[Optional[str]]  = mapped_column(String(50))

class Product(Base):
    __tablename__ = "products"
    productid:    Mapped[int]            = mapped_column(Integer, primary_key=True)
    productname:  Mapped[str]            = mapped_column(String(100))
    unitprice:    Mapped[Optional[float]] = mapped_column(Float)
    unitsinstock: Mapped[Optional[int]]  = mapped_column(Integer)
    discontinued: Mapped[int]            = mapped_column(Integer)

class Order(Base):
    __tablename__ = "orders"
    orderid:      Mapped[int]            = mapped_column(Integer, primary_key=True)
    customerid:   Mapped[Optional[str]]  = mapped_column(String(5),   ForeignKey("customers.customerid"))
    employeeid:   Mapped[Optional[int]]  = mapped_column(Integer,     ForeignKey("employees.employeeid"))
    orderdate:    Mapped[Optional[Date]] = mapped_column(Date)
    requireddate: Mapped[Optional[Date]] = mapped_column(Date)
    shippeddate:  Mapped[Optional[Date]] = mapped_column(Date)
    shipvia:      Mapped[Optional[int]]  = mapped_column(Integer)
    freight:      Mapped[Optional[float]] = mapped_column(Float)
    shipname:     Mapped[Optional[str]]  = mapped_column(String(100))
    shipaddress:  Mapped[Optional[str]]  = mapped_column(String(200))
    shipcity:     Mapped[Optional[str]]  = mapped_column(String(50))
    shipregion:   Mapped[Optional[str]]  = mapped_column(String(50))
    shippostalcode: Mapped[Optional[str]] = mapped_column(String(20))
    shipcountry:  Mapped[Optional[str]]  = mapped_column(String(50))

class OrderDetail(Base):
    __tablename__ = "order_details"
    orderid:   Mapped[int]   = mapped_column(Integer, ForeignKey("orders.orderid"),   primary_key=True)
    productid: Mapped[int]   = mapped_column(Integer, ForeignKey("products.productid"), primary_key=True)
    unitprice: Mapped[float] = mapped_column(Float)
    quantity:  Mapped[int]   = mapped_column(Integer)
    discount:  Mapped[float] = mapped_column(Float)

class Employee(Base):
    __tablename__ = "employees"
    employeeid: Mapped[int]            = mapped_column(Integer, primary_key=True)
    lastname:   Mapped[str]            = mapped_column(String(50))
    firstname:  Mapped[str]            = mapped_column(String(50))
    title:      Mapped[Optional[str]]  = mapped_column(String(100))
    birthdate:  Mapped[Optional[Date]] = mapped_column(Date)
    hiredate:   Mapped[Optional[Date]] = mapped_column(Date)
    # photo y photopath son datos binarios — se excluyen del schema intencionalmente

# Idempotencia: drop en orden inverso de FK, luego create
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
print("✓ Schema creado en PostgreSQL")

# %% [markdown]
# ### Diagrama Entidad-Relación (ERD)
#
# Las 5 tablas de Northwind y sus relaciones:

# %%
HTML("""
<pre class="mermaid" style="background:white;">
erDiagram
    CUSTOMERS {
        string  customerid  PK
        string  companyname
        string  contactname
        string  country
        string  city
    }
    EMPLOYEES {
        int     employeeid  PK
        string  lastname
        string  firstname
        string  title
    }
    PRODUCTS {
        int     productid    PK
        string  productname
        float   unitprice
        int     unitsinstock
        int     discontinued
    }
    ORDERS {
        int     orderid      PK
        string  customerid   FK
        int     employeeid   FK
        date    orderdate
        date    requireddate
        date    shippeddate
        int     shipvia
        float   freight
        string  shipcountry
    }
    ORDER_DETAILS {
        int     orderid    PK,FK
        int     productid  PK,FK
        float   unitprice
        int     quantity
        float   discount
    }

    CUSTOMERS    ||--o{ ORDERS        : "places"
    EMPLOYEES    ||--o{ ORDERS        : "handles"
    ORDERS       ||--|{ ORDER_DETAILS : "contains"
    PRODUCTS     ||--o{ ORDER_DETAILS : "included in"
</pre>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: false });
  await mermaid.run();
</script>
""")

# %% [markdown]
# ### 1.3 Cargar datos Northwind desde CSV
#
# Bootstrap inicial: cargamos los CSVs usando `session.execute(insert(Model), records)`.
# Este es el patrón de bulk insert de SQLAlchemy 2.0 — respeta el schema ORM,
# los tipos de columna y las FK. En producción los datos ya estarían en la base de datos.

# %%
import pandas as pd
from sqlalchemy import insert

def load_csv(session, model, path, parse_dates=None):
    df = pd.read_csv(path, parse_dates=parse_dates)
    # pd.isnull() maneja tanto NaN (float) como NaT (datetime) → None para PostgreSQL
    # df.where(df.notna()) no convierte NaT en columnas datetime — de ahí el error
    records = [
        {k: None if pd.isnull(v) else v for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]
    session.execute(insert(model), records)
    print(f"✓ {model.__tablename__}: {len(records):,} filas cargadas")

# Orden respeta dependencias FK:
#   sin FK: employees, customers, products
#   depende de los anteriores: orders → order_details
with Session(engine) as session:
    load_csv(session, Employee,    "northwind_clean/employees/employees.csv",
             parse_dates=["birthdate", "hiredate"])
    load_csv(session, Customer,    "northwind_clean/customers/customers.csv")
    load_csv(session, Product,     "northwind_clean/products/products.csv")
    load_csv(session, Order,       "northwind_clean/orders/orders.csv",
             parse_dates=["orderdate", "requireddate", "shippeddate"])
    load_csv(session, OrderDetail, "northwind_clean/order_details/order_details.csv")
    session.commit()
    print("✓ Bootstrap completo")

# %% [markdown]
# ### 1.4 CRUD con SQLAlchemy 2.0
#
# > **Nota para data scientists y ML engineers**
# >
# > Dominar CRUD no es opcional cuando construyes productos. En un análisis exploratorio
# > puedes leer datos con `pd.read_sql` y quedarte ahí. Pero cuando construyes un producto
# > de datos o un sistema de ML, estás escribiendo **software** — y ese software necesita
# > crear registros, leer estado, actualizar resultados y eliminar entradas obsoletas.
# >
# > Ejemplos concretos del mundo real:
# > - Un pipeline de ML escribe predicciones en una tabla de resultados (**Create**)
# > - Un dashboard de monitoreo consulta métricas de un modelo en producción (**Read**)
# > - Un retraining job actualiza el estado de un experimento en un registro (**Update**)
# > - Un sistema de features elimina features deprecadas del feature store (**Delete**)
# >
# > La diferencia entre un data scientist que analiza datos y uno que construye productos
# > está, en gran parte, en saber hacer esto bien.
#
# CRUD son las cuatro operaciones fundamentales de cualquier base de datos relacional.
# En SQLAlchemy 2.0 todas se ejecutan dentro de una **Session**.
#
# La `Session` es el corazón del ORM: actúa como una "zona de trabajo" que:
# - Rastrea qué objetos Python fueron creados, modificados o eliminados
# - Agrupa esos cambios en una transacción
# - Los envía a la base de datos cuando llamas `session.commit()`
# - Los deshace si ocurre un error (`session.rollback()`)
#
# El bloque `with Session(engine) as session:` garantiza que la sesión se cierra
# correctamente al terminar, incluso si hay una excepción.

# %% [markdown]
# #### Create — INSERT
#
# Crear un objeto Python de la clase `Customer` es equivalente a preparar un INSERT.
# El objeto aún no existe en la base de datos — solo existe en memoria.
#
# `session.add(nuevo_cliente)` registra el objeto en la sesión (lo marca como "pendiente").
# `session.commit()` envía el INSERT a PostgreSQL y confirma la transacción.

# %%
with Session(engine) as session:
    nuevo_cliente = Customer(
        customerid  = "ITAM1",
        companyname = "ITAM Business School",
        country     = "Mexico",
        city        = "Ciudad de Mexico"
    )
    session.add(nuevo_cliente)
    session.commit()
    print(f"✓ Create: cliente '{nuevo_cliente.customerid}' creado")

# %% [markdown]
# ```sql
# -- SQL equivalente generado por SQLAlchemy:
# INSERT INTO customers (customerid, companyname, country, city)
# VALUES ('ITAM1', 'ITAM Business School', 'Mexico', 'Ciudad de Mexico')
# ```

# %% [markdown]
# #### Read — SELECT
#
# `select(Product)` construye una query SELECT sobre la tabla `products`.
# `.where(...)` agrega la cláusula WHERE. `.order_by(...)` agrega ORDER BY.
# La query todavía no se ha ejecutado — SQLAlchemy la construye como un objeto.
#
# `session.execute(stmt)` envía la query a PostgreSQL y devuelve un `Result`.
# `.scalars()` extrae los objetos ORM (en lugar de tuplas).
# `.all()` materializa el resultado como una lista de objetos `Product`.
#
# Cada elemento de `productos` es una instancia de `Product` — puedes acceder
# a sus columnas como atributos Python: `p.productname`, `p.unitprice`.

# %%
with Session(engine) as session:
    stmt = select(Product).where(Product.unitprice > 20).order_by(Product.unitprice.desc())
    productos = session.execute(stmt).scalars().all()
    print(f"✓ Read: {len(productos)} productos con precio > $20")
    for p in productos[:5]:
        print(f"   {p.productname}: ${p.unitprice}")

# %% [markdown]
# ```sql
# -- SQL equivalente generado por SQLAlchemy:
# SELECT * FROM products WHERE unitprice > 20 ORDER BY unitprice DESC
# ```

# %% [markdown]
# #### Update — UPDATE
#
# `session.get(Product, 1)` busca el registro por primary key — es equivalente a
# `SELECT * FROM products WHERE productid = 1`. Devuelve el objeto directamente.
#
# Una vez que el objeto está dentro de la sesión, SQLAlchemy lo "observa".
# Al modificar un atributo (`producto.unitprice = 19.99`), la sesión detecta
# el cambio automáticamente — esto se llama **Unit of Work**.
#
# `session.commit()` genera el UPDATE solo para las columnas que cambiaron
# y lo envía a PostgreSQL.

# %%
with Session(engine) as session:
    producto = session.get(Product, 1)
    precio_anterior = producto.unitprice
    producto.unitprice = 19.99
    session.commit()
    print(f"✓ Update: producto 1 — ${precio_anterior} → ${producto.unitprice}")

# %% [markdown]
# ```sql
# -- SQL equivalente generado por SQLAlchemy:
# UPDATE products SET unitprice = 19.99 WHERE productid = 1
# ```

# %% [markdown]
# #### Delete — DELETE
#
# `session.get(Customer, "ITAM1")` recupera el cliente por primary key.
# El `if cliente:` protege contra el caso en que el cliente no exista
# (por ejemplo, si esta celda se corre dos veces).
#
# `session.delete(cliente)` marca el objeto para eliminación dentro de la sesión.
# `session.commit()` ejecuta el DELETE en PostgreSQL.
#
# Nota: si existieran órdenes asociadas a "ITAM1", PostgreSQL lanzaría un error
# de FK constraint — no puedes eliminar un cliente que tiene órdenes activas.
# En este caso el cliente es nuevo y no tiene órdenes, por eso el DELETE funciona.

# %%
with Session(engine) as session:
    cliente = session.get(Customer, "ITAM1")
    if cliente:
        session.delete(cliente)
        session.commit()
        print(f"✓ Delete: cliente '{cliente.customerid}' eliminado")

# %% [markdown]
# ```sql
# -- SQL equivalente generado por SQLAlchemy:
# DELETE FROM customers WHERE customerid = 'ITAM1'
# ```

# %% [markdown]
# ---
# ## Parte 2: ETL — Bronze Layer (Raw)
#
# ### 2.1 Extract desde la Read Replica
#
# El ETL conecta a la **Read Replica** — nunca a la primaria — para evitar contención
# con la aplicación productiva. Solo cambia el endpoint; el código es idéntico.

# %%
# Read Replica — endpoint separado de la primaria
REPLICA_ENDPOINT = "northwind-replica.xxxx.us-east-1.rds.amazonaws.com"

engine_etl = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{REPLICA_ENDPOINT}:{DB_PORT}/{DB_NAME}"
)

df_orders        = pd.read_sql("SELECT * FROM orders",        engine_etl)
df_order_details = pd.read_sql("SELECT * FROM order_details", engine_etl)
df_products      = pd.read_sql("SELECT * FROM products",      engine_etl)
df_customers     = pd.read_sql("SELECT * FROM customers",     engine_etl)
df_employees     = pd.read_sql("SELECT * FROM employees",     engine_etl)

print(f"✓ orders:        {len(df_orders):,} filas")
print(f"✓ order_details: {len(df_order_details):,} filas")
print(f"✓ products:      {len(df_products):,} filas")
print(f"✓ customers:     {len(df_customers):,} filas")
print(f"✓ employees:     {len(df_employees):,} filas")

# %% [markdown]
# ### 2.2 Load a Bronze (mode="overwrite")
#
# Datos sin transformar. `mode="overwrite"` garantiza idempotencia:
# correr el pipeline dos veces no duplica datos.

# %%
import awswrangler as wr

BUCKET_NAME = "itam-analytics-dante"  # OJO: cambia por tu nombre

wr.catalog.create_database(name="northwind_bronze", exist_ok=True)

def load_bronze(df, table_name):
    wr.s3.to_parquet(
        df       = df,
        path     = f"s3://{BUCKET_NAME}/northwind/bronze/{table_name}/",
        dataset  = True,
        database = "northwind_bronze",
        table    = table_name,
        mode     = "overwrite",
    )
    print(f"✓ Bronze/{table_name}: {len(df):,} filas")

load_bronze(df_orders,        "orders")
load_bronze(df_order_details, "order_details")
load_bronze(df_products,      "products")
load_bronze(df_customers,     "customers")
load_bronze(df_employees,     "employees")

# %% [markdown]
# ---
# ## Parte 3: ETL — Silver Layer (Staging)
#
# ### 3.1 Transform
#
# Aplicamos las transformaciones que garantizan datos confiables:
# tipos correctos, nulls manejados, validaciones con assert.

# %%
import numpy as np

# --- orders ---
df_orders_silver = (
    df_orders
    .assign(
        orderdate   = lambda df_: pd.to_datetime(df_["orderdate"],   errors="coerce"),
        shippeddate = lambda df_: pd.to_datetime(df_["shippeddate"], errors="coerce"),
        year        = lambda df_: df_["orderdate"].dt.year.astype("Int64"),
    )
)

assert len(df_orders_silver) > 0,                      "orders está vacío"
assert df_orders_silver["orderid"].notna().all(),      "orderid tiene nulos"
assert df_orders_silver["orderdate"].notna().any(),    "orderdate todas nulas"
assert df_orders_silver.duplicated(subset=["orderid"]).sum() == 0, "orderid duplicados"
print(f"✓ orders silver: {len(df_orders_silver):,} filas | {df_orders_silver['year'].nunique()} años")

# %%
# --- products ---
df_products_silver = (
    df_products
    .assign(
        unitprice    = lambda df_: pd.to_numeric(df_["unitprice"],    errors="coerce"),
        unitsinstock = lambda df_: pd.to_numeric(df_["unitsinstock"], errors="coerce"),
        discontinued = lambda df_: df_["discontinued"].astype(int),
    )
)

assert len(df_products_silver) > 0, "products está vacío"
print(f"✓ products silver: {len(df_products_silver):,} filas")

# %%
# --- order_details ---
df_order_details_silver = (
    df_order_details
    .assign(
        unitprice = lambda df_: pd.to_numeric(df_["unitprice"], errors="coerce"),
        quantity  = lambda df_: pd.to_numeric(df_["quantity"],  errors="coerce"),
        discount  = lambda df_: pd.to_numeric(df_["discount"],  errors="coerce"),
    )
)

assert len(df_order_details_silver) > 0, "order_details está vacío"
print(f"✓ order_details silver: {len(df_order_details_silver):,} filas")

# %%
# --- customers ---
df_customers_silver = df_customers.copy()
assert len(df_customers_silver) > 0, "customers está vacío"
print(f"✓ customers silver: {len(df_customers_silver):,} filas")

# %%
# --- employees: convertir fechas (photo/photopath no están en el modelo ORM) ---
df_employees_silver = (
    df_employees
    .assign(
        birthdate = lambda df_: pd.to_datetime(df_["birthdate"], errors="coerce"),
        hiredate  = lambda df_: pd.to_datetime(df_["hiredate"],  errors="coerce"),
    )
)

assert len(df_employees_silver) > 0, "employees está vacío"
print(f"✓ employees silver: {len(df_employees_silver):,} filas")

# %% [markdown]
# ### 3.2 Load a Silver
#
# Parquet + Snappy + particionamiento por año para orders.
# `mode="overwrite_partitions"` para orders: solo reescribe las particiones afectadas.

# %%
wr.catalog.create_database(name="northwind_silver", exist_ok=True)

# Tablas sin partición → overwrite completo
for df, name in [
    (df_products_silver,      "products"),
    (df_customers_silver,     "customers"),
    (df_employees_silver,     "employees"),
    (df_order_details_silver, "order_details"),
]:
    wr.s3.to_parquet(
        df          = df,
        path        = f"s3://{BUCKET_NAME}/northwind/silver/{name}/",
        dataset     = True,
        database    = "northwind_silver",
        table       = name,
        compression = "snappy",
        mode        = "overwrite",
    )
    print(f"✓ Silver/{name}: {len(df):,} filas")

# %%
# orders particionada por año → overwrite_partitions
wr.s3.to_parquet(
    df             = df_orders_silver,
    path           = f"s3://{BUCKET_NAME}/northwind/silver/orders/",
    dataset        = True,
    database       = "northwind_silver",
    table          = "orders",
    partition_cols = ["year"],
    compression    = "snappy",
    mode           = "overwrite_partitions",
)
print(f"✓ Silver/orders: {len(df_orders_silver):,} filas (particionado por year)")

# %% [markdown]
# ---
# ## Parte 4: ELT — Gold Layer (Analytics)
#
# ### 4.1 CTAS en Athena: tabla ventas
#
# Construimos la tabla Gold con un JOIN de 4 tablas Silver.
# Incluimos `employeeid` para poder hacer el join con employees en Athena.

# %%
# Idempotencia: eliminar la tabla si ya existe antes de recrearla
wr.catalog.delete_table_if_exists(database="northwind_gold", table="ventas")
wr.catalog.create_database(name="northwind_gold", exist_ok=True)

query_ctas = f"""
CREATE TABLE northwind_gold.ventas
WITH (
    format           = 'PARQUET',
    write_compression = 'SNAPPY',
    external_location = 's3://{BUCKET_NAME}/northwind/gold/ventas/'
) AS (
    SELECT
        o.orderid,
        o.orderdate,
        o.employeeid,
        c.country,
        c.companyname,
        p.productname,
        ROUND(od.unitprice * od.quantity * (1 - od.discount), 2) AS revenue
    FROM northwind_silver.orders o
    JOIN northwind_silver.order_details od ON o.orderid    = od.orderid
    JOIN northwind_silver.customers     c  ON o.customerid = c.customerid
    JOIN northwind_silver.products      p  ON od.productid = p.productid
)
"""

wr.athena.read_sql_query(
    query_ctas,
    database      = "northwind_gold",
    ctas_approach = False,
)
print("✓ Gold/ventas creada")

# %% [markdown]
# Verificar la tabla Gold:

# %%
df_gold_sample = wr.athena.read_sql_query(
    "SELECT * FROM northwind_gold.ventas LIMIT 5",
    database      = "northwind_gold",
    ctas_approach = False,
)
df_gold_sample

# %% [markdown]
# ---
# ## Parte 5: Analítica con Athena
#
# ### Q1 — Revenue total por país

# %%
query_q1 = """
SELECT
    country,
    COUNT(DISTINCT orderid)                  AS num_ordenes,
    ROUND(SUM(revenue), 2)                   AS revenue_total,
    RANK() OVER (ORDER BY SUM(revenue) DESC) AS ranking
FROM northwind_gold.ventas
GROUP BY country
ORDER BY revenue_total DESC
"""

df_q1 = wr.athena.read_sql_query(query_q1, database="northwind_gold", ctas_approach=False)
df_q1

# %% [markdown]
# ### Q2 — Top 10 productos por revenue

# %%
query_q2 = """
WITH producto_revenue AS (
    SELECT
        productname,
        ROUND(SUM(revenue), 2) AS revenue_total
    FROM northwind_gold.ventas
    GROUP BY productname
)
SELECT productname, revenue_total
FROM producto_revenue
ORDER BY revenue_total DESC
LIMIT 10
"""

df_q2 = wr.athena.read_sql_query(query_q2, database="northwind_gold", ctas_approach=False)
df_q2

# %% [markdown]
# ### Q3 — Tendencia mensual con variación vs mes anterior

# %%
query_q3 = """
WITH mensual AS (
    SELECT
        DATE_FORMAT(orderdate, '%Y-%m')  AS mes,
        ROUND(SUM(revenue), 2)           AS revenue
    FROM northwind_gold.ventas
    GROUP BY 1
)
SELECT
    mes,
    revenue,
    LAG(revenue) OVER (ORDER BY mes)                                      AS revenue_mes_anterior,
    ROUND(revenue - LAG(revenue) OVER (ORDER BY mes), 2)                  AS variacion,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY mes))
          / NULLIF(LAG(revenue) OVER (ORDER BY mes), 0), 1)               AS variacion_pct
FROM mensual
ORDER BY mes
"""

df_q3 = wr.athena.read_sql_query(query_q3, database="northwind_gold", ctas_approach=False)
df_q3

# %% [markdown]
# ### Q4 — Empleado con más ventas
#
# JOIN entre Gold (ventas) y Silver (employees).
# Gold no tiene que incluirlo todo — las capas se pueden combinar.

# %%
query_q4 = """
SELECT
    e.firstname || ' ' || e.lastname   AS empleado,
    COUNT(DISTINCT v.orderid)          AS num_ordenes,
    ROUND(SUM(v.revenue), 2)           AS revenue_total
FROM northwind_gold.ventas v
JOIN northwind_silver.employees e ON v.employeeid = e.employeeid
GROUP BY e.firstname, e.lastname
ORDER BY revenue_total DESC
"""

df_q4 = wr.athena.read_sql_query(query_q4, database="northwind_gold", ctas_approach=False)
df_q4

# %% [markdown]
# ### Visualizaciones

# %%
import matplotlib.pyplot as plt

plt.style.use("ggplot")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Revenue por país (top 10)
df_q1.head(10).plot(
    kind    = "barh",
    x       = "country",
    y       = "revenue_total",
    ax      = axes[0],
    color   = "#1976d2",
    legend  = False,
)
axes[0].set_title("Revenue total por país (Top 10)")
axes[0].set_xlabel("Revenue (USD)")
axes[0].set_ylabel("País")
axes[0].invert_yaxis()

# Tendencia mensual
df_q3.plot(
    kind    = "line",
    x       = "mes",
    y       = "revenue",
    ax      = axes[1],
    color   = "#1976d2",
    marker  = "o",
    legend  = False,
)
axes[1].set_title("Tendencia mensual de revenue")
axes[1].set_xlabel("Mes")
axes[1].set_ylabel("Revenue (USD)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()

import os
os.makedirs("figuras", exist_ok=True)
plt.savefig("figuras/revenue_analytics.png", dpi=300, bbox_inches="tight")
plt.show()
