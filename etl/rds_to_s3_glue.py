"""
ETL: Exporta tablas de RDS PostgreSQL a S3 (Parquet) y registra en Glue Data Catalog.
Capa analitica Gold del data lake -- complementaria a RDS (OLTP).
"""

import json, sys, boto3, psycopg2, awswrangler as wr, pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

SECRET_NAME = "rds/retail-poc-credentials"
AWS_REGION = "us-east-1"
S3_BUCKET = "1c-retail-poc-334931733619"
GLUE_DB = "retail_poc"

TABLES = [
    "products",
    "predictions",
    "actuals",
    "evaluation_metrics",
    "business_feedback",
    "flagged_products",
]


def get_rds_connection():
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    secret = json.loads(client.get_secret_value(SecretId=SECRET_NAME)["SecretString"])
    return psycopg2.connect(
        host=secret["host"],
        port=int(secret.get("port", 5432)),
        dbname=secret["dbname"],
        user=secret["username"],
        password=secret["password"],
        sslmode="require",
        connect_timeout=10,
    )


def fix_null_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas object con todos los valores null a string para awswrangler."""
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].isna().all():
            df[col] = df[col].astype('string')
    return df


def export_table(conn, table_name: str) -> None:
    s3_path = f"s3://{S3_BUCKET}/gold/{table_name}/"
    print(f"\nExportando {table_name} -> {s3_path}")

    df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
    df = fix_null_columns(df)
    print(f"   Filas: {len(df):,}  Columnas: {list(df.columns)}")

    wr.s3.to_parquet(
        df=df,
        path=s3_path,
        dataset=True,
        database=GLUE_DB,
        table=table_name,
        mode="overwrite",
        compression="snappy",
    )
    print(f"   OK Registrado en Glue: {GLUE_DB}.{table_name}")


def main():
    print("=" * 60)
    print("ETL RDS -> S3 (Parquet) + Glue Data Catalog")
    print("=" * 60)

    conn = get_rds_connection()
    for table in TABLES:
        export_table(conn, table)

    s3_mv = f"s3://{S3_BUCKET}/gold/mv_app_data/"
    print(f"\nExportando mv_app_data (vista materializada) -> {s3_mv}")
    df_mv = pd.read_sql('SELECT * FROM mv_app_data', conn)
    df_mv = fix_null_columns(df_mv)
    print(f"   Filas: {len(df_mv):,}")
    wr.s3.to_parquet(
        df=df_mv,
        path=s3_mv,
        dataset=True,
        database=GLUE_DB,
        table="mv_app_data",
        mode="overwrite",
        compression="snappy",
    )
    print(f"   OK Registrado en Glue: {GLUE_DB}.mv_app_data")

    conn.close()
    print("\n" + "=" * 60)
    print("Done. Todas las tablas estan en S3 + Glue Data Catalog.")
    print("=" * 60)


if __name__ == "__main__":
    main()
