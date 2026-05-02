import os
import json
import boto3
import psycopg2
import streamlit as st

SECRET_NAME = "rds/1c-credentials"
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


@st.cache_resource
def get_connection():
    """Conexión a RDS vía Secrets Manager. El host se lee del secret (campo 'host')
    o del env var RDS_ENDPOINT si el secret no lo incluye todavía."""
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    secret = json.loads(
        client.get_secret_value(SecretId=SECRET_NAME)["SecretString"]
    )
    host = secret.get("host") or os.environ["RDS_ENDPOINT"]
    return psycopg2.connect(
        host=host,
        port=int(secret.get("port", 5432)),
        dbname=secret["dbname"],
        user=secret["username"],
        password=secret["password"],
        connect_timeout=5,
    )


def query(sql: str, params=None) -> list[dict]:
    """SELECT → lista de dicts (una fila = un dict con nombres de columna como keys)."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def execute(sql: str, params=None):
    """INSERT / UPDATE / DELETE."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()


def execute_returning_id(sql: str, params=None) -> str | None:
    """INSERT con RETURNING id → UUID como string."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql, params)
        result = cur.fetchone()
    conn.commit()
    return str(result[0]) if result else None
