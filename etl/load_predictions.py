"""
ETL — Carga predicciones de demanda a RDS.

Descarga datos de S3, construye features, genera predicciones para Nov-2015
(date_block_num=34) e inserta en: products, actuals (Oct-2015), predictions.

Env vars requeridas:
  S3_BUCKET        — bucket donde viven los artefactos

Opcionales:
  S3_RAW_PREFIX    — prefijo de los CSVs crudos        (default: raw/)
  S3_MODEL_KEY     — key del model.joblib               (default: models/model.joblib)
  SECRET_ID        — nombre del secret en Secrets Mgr   (default: rds/1c-credentials)
  AWS_DEFAULT_REGION                                    (default: us-east-1)
  RDS_ENDPOINT     — fallback si el secret no trae host

Ejecución:
  python etl/load_predictions.py
"""

import json
import os
import tempfile
from itertools import product as iproduct
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

# ── Configuración ──────────────────────────────────────────────────────────────
S3_BUCKET     = os.environ["S3_BUCKET"]
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX", "raw/")
S3_MODEL_KEY  = os.getenv("S3_MODEL_KEY",  "models/model.joblib")
SECRET_ID     = os.getenv("SECRET_ID",     "rds/1c-credentials")
AWS_REGION    = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

VALID_MONTH  = 33          # Oct 2015 — ground truth
TEST_MONTH   = 34          # Nov 2015 — predicciones a publicar
ACTUAL_DATE  = "2015-10-01"
PRED_DATE    = "2015-11-01"
CLIP_MIN, CLIP_MAX = 0, 20

FEATURE_COLS = [
    "shop_id", "item_id", "item_category_id", "month", "year", "shop_size",
    "item_cnt_month_lag_1",  "item_price_mean_lag_1",
    "item_cnt_month_lag_2",  "item_price_mean_lag_2",
    "item_cnt_month_lag_3",  "item_price_mean_lag_3",
    "item_cnt_month_lag_6",  "item_price_mean_lag_6",
    "item_cnt_month_lag_12", "item_price_mean_lag_12",
    "shop_mean_lag_1", "item_mean_lag_1", "cat_mean_lag_1",
]


# ── Conexión RDS vía Secrets Manager ──────────────────────────────────────────
def get_connection():
    client = boto3.client("secretsmanager", region_name=AWS_REGION)
    secret = json.loads(
        client.get_secret_value(SecretId=SECRET_ID)["SecretString"]
    )
    host = secret.get("host") or os.environ["RDS_ENDPOINT"]
    return psycopg2.connect(
        host=host,
        port=int(secret.get("port", 5432)),
        dbname=secret["dbname"],
        user=secret["username"],
        password=secret["password"],
    )


# ── Descarga de artefactos desde S3 ───────────────────────────────────────────
def download_s3(tmp: Path):
    s3  = boto3.client("s3", region_name=AWS_REGION)
    pfx = S3_RAW_PREFIX.rstrip("/") + "/"
    for fname in ["items.csv", "sales_train.csv", "test.csv"]:
        s3.download_file(S3_BUCKET, pfx + fname, str(tmp / fname))
        print(f"  s3 → {fname}")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(tmp / "model.joblib"))
    print("  s3 → model.joblib")


# ── Feature engineering (replica de prep.py) ──────────────────────────────────
def build_matrix(tmp: Path) -> pd.DataFrame:
    items = pd.read_csv(tmp / "items.csv")
    sales = pd.read_csv(tmp / "sales_train.csv")
    test  = pd.read_csv(tmp / "test.csv")

    # Limpieza básica
    sales = sales[(sales["item_price"] > 0) & (sales["item_cnt_day"] >= 0)].copy()

    # Agregación mensual
    KEY = ["date_block_num", "shop_id", "item_id"]
    monthly = sales.groupby(KEY, as_index=False).agg(
        item_cnt_month=("item_cnt_day",  "sum"),
        item_price_mean=("item_price",   "mean"),
    )

    # shop_size por percentiles del total histórico
    totals   = sales.groupby("shop_id")["item_cnt_day"].sum()
    p33, p66 = totals.quantile(0.33), totals.quantile(0.66)
    smap     = pd.Series("average", index=totals.index)
    smap[totals <= p33] = "small"
    smap[totals >  p66] = "large"
    monthly["shop_size"] = monthly["shop_id"].map(smap)

    # Grid: combinaciones observadas por mes (igual que prep.py)
    first_m = int(sales["date_block_num"].min())
    last_m  = int(sales["date_block_num"].max())   # 33
    grids   = []
    for m in range(first_m, last_m + 1):
        sm = sales[sales["date_block_num"] == m]
        grids.append(
            np.array(
                list(iproduct([m], sm["shop_id"].unique(), sm["item_id"].unique())),
                dtype=np.int32,
            )
        )
    matrix = pd.DataFrame(np.vstack(grids), columns=KEY)

    # Agregar mes de test (34) con los pares de test.csv
    tp = test[["shop_id", "item_id"]].copy()
    tp["date_block_num"] = TEST_MONTH
    matrix = pd.concat([matrix, tp[KEY]], ignore_index=True)

    # Merge con agregados mensuales
    matrix = matrix.merge(monthly, on=KEY, how="left")

    # Features base
    enc = {"small": 0, "average": 1, "large": 2}
    matrix["shop_size"]       = matrix["shop_size"].fillna("average").map(enc).astype(np.int8)
    matrix["item_cnt_month"]  = matrix["item_cnt_month"].fillna(0).clip(0, 20).astype(np.float32)
    matrix["item_price_mean"] = matrix["item_price_mean"].fillna(0).astype(np.float32)

    cat_map = items.set_index("item_id")["item_category_id"]
    matrix["item_category_id"] = matrix["item_id"].map(cat_map).astype(np.int16)
    matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
    matrix["year"]  = (2013 + matrix["date_block_num"] // 12).astype(np.int16)

    # Rezagos
    for lag in [1, 2, 3, 6, 12]:
        src = matrix[KEY + ["item_cnt_month", "item_price_mean"]].copy()
        src["date_block_num"] = src["date_block_num"] + lag
        src = src.rename(columns={
            "item_cnt_month":  f"item_cnt_month_lag_{lag}",
            "item_price_mean": f"item_price_mean_lag_{lag}",
        })
        matrix = matrix.merge(src, on=KEY, how="left")

    # Promedios grupales con lag 1
    for gcols, fname in [
        (["shop_id"],          "shop_mean_lag_1"),
        (["item_id"],          "item_mean_lag_1"),
        (["item_category_id"], "cat_mean_lag_1"),
    ]:
        mc  = ["date_block_num"] + gcols
        grp = (
            matrix.groupby(mc, as_index=False)["item_cnt_month"]
            .mean()
            .rename(columns={"item_cnt_month": fname})
        )
        grp["date_block_num"] = grp["date_block_num"] + 1
        matrix = matrix.merge(grp, on=mc, how="left")

    # Relleno final de lags (fillna 0, igual que prep.py)
    lag_cols = list(dict.fromkeys(
        [c for c in matrix.columns if "lag_" in c]
        + ["shop_mean_lag_1", "item_mean_lag_1", "cat_mean_lag_1"]
    ))
    matrix[lag_cols] = matrix[lag_cols].fillna(0).astype(np.float32)

    return matrix


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=== load_predictions.py ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        print("1) Descargando artefactos de S3 …")
        download_s3(tmp)

        print("2) Construyendo matriz de features …")
        matrix = build_matrix(tmp)
        print(f"   {matrix.shape[0]:,} filas en la matriz completa")

        print("3) Cargando modelo …")
        model = joblib.load(tmp / "model.joblib")

        # RMSE en mes de validación (33) para intervalo de confianza ±1.5×RMSE
        valid  = matrix[matrix["date_block_num"] == VALID_MONTH].copy()
        pv     = np.clip(model.predict(valid[FEATURE_COLS]), CLIP_MIN, CLIP_MAX)
        rmse   = float(np.sqrt(np.mean((pv - valid["item_cnt_month"].values) ** 2)))
        ci     = 1.5 * rmse
        print(f"   RMSE mes {VALID_MONTH}: {rmse:.4f}  →  CI ±{ci:.4f}")

        # Predicciones mes 34
        test_df = matrix[matrix["date_block_num"] == TEST_MONTH].copy()
        preds   = np.clip(model.predict(test_df[FEATURE_COLS]), CLIP_MIN, CLIP_MAX)
        test_df["predicted_sales"] = preds
        test_df["lower_bound"]     = np.clip(preds - ci, CLIP_MIN, None)
        test_df["upper_bound"]     = preds + ci   # sin techo — fuera del rango [0,20] es OK en upper

        print("4) Conectando a RDS …")
        conn = get_connection()
        try:
            cur = conn.cursor()

            # ── products ──────────────────────────────────────────────────────
            all_pairs = (
                pd.concat([
                    valid[["item_id", "shop_id", "item_category_id"]],
                    test_df[["item_id", "shop_id", "item_category_id"]],
                ])
                .drop_duplicates(["item_id", "shop_id"])
            )
            prod_rows = [
                (str(int(r.item_id)), str(int(r.item_category_id)), str(int(r.shop_id)))
                for r in all_pairs.itertuples()
            ]
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO products (item_id, category, shop_id) VALUES %s "
                "ON CONFLICT (item_id, shop_id) DO NOTHING",
                prod_rows,
            )
            print(f"   products: {len(prod_rows)} pares procesados (sin duplicados)")

            # Recuperar UUIDs para el join posterior
            cur.execute("SELECT id, item_id, shop_id FROM products")
            uuid_map = {(r[1], r[2]): r[0] for r in cur.fetchall()}

            # ── actuals (mes 33) ──────────────────────────────────────────────
            cur.execute("DELETE FROM actuals WHERE sale_date = %s", (ACTUAL_DATE,))
            act_rows = [
                (
                    uuid_map[(str(int(r.item_id)), str(int(r.shop_id)))],
                    ACTUAL_DATE,
                    float(r.item_cnt_month),
                )
                for r in valid.itertuples()
                if (str(int(r.item_id)), str(int(r.shop_id))) in uuid_map
            ]
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO actuals (product_id, sale_date, actual_sales) VALUES %s",
                act_rows,
            )
            print(f"   actuals: {len(act_rows)} filas")

            # ── predictions (mes 34) ──────────────────────────────────────────
            cur.execute("DELETE FROM predictions WHERE prediction_date = %s", (PRED_DATE,))
            pred_rows_db = [
                (
                    uuid_map[(str(int(r.item_id)), str(int(r.shop_id)))],
                    PRED_DATE,
                    float(r.predicted_sales),
                    float(r.lower_bound),
                    float(r.upper_bound),
                )
                for r in test_df.itertuples()
                if (str(int(r.item_id)), str(int(r.shop_id))) in uuid_map
            ]
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO predictions "
                "(product_id, prediction_date, predicted_sales, lower_bound, upper_bound) "
                "VALUES %s",
                pred_rows_db,
            )
            print(f"   predictions: {len(pred_rows_db)} filas")

            conn.commit()
            print("✓ commit OK")
        finally:
            conn.close()

    print("=== done ===")


if __name__ == "__main__":
    main()
