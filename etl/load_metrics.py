"""
ETL — Calcula métricas de evaluación del modelo vs naive y las carga a RDS.

Descarga el modelo desde S3, reconstruye la matriz de features para el mes de
validación (33 = Oct-2015), calcula RMSE/MAE globales y por grupo
(categoría, tienda), compara contra un baseline naive (lag_1 = mes 32),
e inserta todo en la tabla evaluation_metrics.

Env vars requeridas:
  S3_BUCKET        — bucket donde viven los artefactos

Opcionales:
  S3_RAW_PREFIX    — prefijo de los CSVs crudos        (default: raw/)
  S3_MODEL_KEY     — key del model.joblib               (default: models/model.joblib)
  SECRET_ID        — nombre del secret en Secrets Mgr   (default: rds/1c-credentials)
  AWS_DEFAULT_REGION                                    (default: us-east-1)
  RDS_ENDPOINT     — fallback si el secret no trae host

Ejecución:
  python etl/load_metrics.py
"""

import json
import os
import tempfile
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
CLIP_MIN, CLIP_MAX = 0, 20

FEATURE_COLS = [
    "shop_id", "item_id", "item_category_id", "cat_lvl1", "cat_lvl2",
    "month", "year", "shop_size",
    "item_cnt_month_lag_1",  "item_price_mean_lag_1",
    "item_cnt_month_lag_2",  "item_price_mean_lag_2",
    "item_cnt_month_lag_3",  "item_price_mean_lag_3",
    "item_cnt_month_lag_6",  "item_price_mean_lag_6",
    "item_cnt_month_lag_12", "item_price_mean_lag_12",
    "shop_mean_lag_1", "item_mean_lag_1", "cat_mean_lag_1",
    "cat_lvl1_mean_lag_1", "cat_lvl2_mean_lag_1",
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
    for fname in ["items.csv", "item_categories.csv", "sales_train.csv", "test.csv"]:
        s3.download_file(S3_BUCKET, pfx + fname, str(tmp / fname))
        print(f"  s3 → {fname}")
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(tmp / "model.joblib"))
    print("  s3 → model.joblib")


# ── Feature engineering (replica exacta de load_predictions.py) ───────────────
def _preparar_items_con_categorias(
    items: pd.DataFrame, item_categories: pd.DataFrame
) -> pd.DataFrame:
    """Agrega niveles de categoría a items desde item_categories.csv."""
    tbl_categories = item_categories.copy()
    partes_categoria = tbl_categories["item_category_name"].str.split("-", n=1, expand=True)
    tbl_categories["cat_lvl1_name"] = partes_categoria[0].str.strip()
    tbl_categories["cat_lvl2_name"] = partes_categoria[1].fillna("unknown").str.strip()
    tbl_categories["cat_lvl1"] = (
        tbl_categories["cat_lvl1_name"].astype("category").cat.codes.astype(np.int8)
    )
    tbl_categories["cat_lvl2"] = (
        tbl_categories["cat_lvl2_name"].astype("category").cat.codes.astype(np.int8)
    )
    tbl_categories = tbl_categories[["item_category_id", "cat_lvl1", "cat_lvl2"]].copy()
    resultado = items.merge(tbl_categories, on="item_category_id", how="left")
    resultado["cat_lvl1"] = resultado["cat_lvl1"].fillna(-1).astype(np.int8)
    resultado["cat_lvl2"] = resultado["cat_lvl2"].fillna(-1).astype(np.int8)
    return resultado


def build_matrix(tmp: Path) -> pd.DataFrame:
    from itertools import product as iproduct

    items = pd.read_csv(tmp / "items.csv")
    item_categories = pd.read_csv(tmp / "item_categories.csv")
    sales = pd.read_csv(tmp / "sales_train.csv")
    test  = pd.read_csv(tmp / "test.csv")

    items = _preparar_items_con_categorias(items, item_categories)

    sales = sales[(sales["item_price"] > 0) & (sales["item_cnt_day"] >= 0)].copy()

    KEY = ["date_block_num", "shop_id", "item_id"]
    monthly = sales.groupby(KEY, as_index=False).agg(
        item_cnt_month=("item_cnt_day",  "sum"),
        item_price_mean=("item_price",   "mean"),
    )

    totals   = sales.groupby("shop_id")["item_cnt_day"].sum()
    p33, p66 = totals.quantile(0.33), totals.quantile(0.66)
    smap     = pd.Series("average", index=totals.index)
    smap[totals <= p33] = "small"
    smap[totals >  p66] = "large"
    monthly["shop_size"] = monthly["shop_id"].map(smap)

    first_m = int(sales["date_block_num"].min())
    last_m  = int(sales["date_block_num"].max())
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
    tp["date_block_num"] = 34
    matrix = pd.concat([matrix, tp[KEY]], ignore_index=True)

    matrix = matrix.merge(monthly, on=KEY, how="left")

    enc = {"small": 0, "average": 1, "large": 2}
    matrix["shop_size"]       = matrix["shop_size"].fillna("average").map(enc).astype(np.int8)
    matrix["item_cnt_month"]  = matrix["item_cnt_month"].fillna(0).clip(0, 20).astype(np.float32)
    matrix["item_price_mean"] = matrix["item_price_mean"].fillna(0).astype(np.float32)

    item_feature_map = items.set_index("item_id")[["item_category_id", "cat_lvl1", "cat_lvl2"]]
    matrix = matrix.merge(item_feature_map, left_on="item_id", right_index=True, how="left")
    matrix["item_category_id"] = matrix["item_category_id"].fillna(-1).astype(np.int16)
    matrix["cat_lvl1"] = matrix["cat_lvl1"].fillna(-1).astype(np.int8)
    matrix["cat_lvl2"] = matrix["cat_lvl2"].fillna(-1).astype(np.int8)
    matrix["month"] = (matrix["date_block_num"] % 12).astype(np.int8)
    matrix["year"]  = (2013 + matrix["date_block_num"] // 12).astype(np.int16)

    for lag in [1, 2, 3, 6, 12]:
        src = matrix[KEY + ["item_cnt_month", "item_price_mean"]].copy()
        src["date_block_num"] = src["date_block_num"] + lag
        src = src.rename(columns={
            "item_cnt_month":  f"item_cnt_month_lag_{lag}",
            "item_price_mean": f"item_price_mean_lag_{lag}",
        })
        matrix = matrix.merge(src, on=KEY, how="left")

    for gcols, fname in [
        (["shop_id"],          "shop_mean_lag_1"),
        (["item_id"],          "item_mean_lag_1"),
        (["item_category_id"], "cat_mean_lag_1"),
        (["cat_lvl1"],         "cat_lvl1_mean_lag_1"),
        (["cat_lvl2"],         "cat_lvl2_mean_lag_1"),
    ]:
        mc  = ["date_block_num"] + gcols
        grp = (
            matrix.groupby(mc, as_index=False)["item_cnt_month"]
            .mean()
            .rename(columns={"item_cnt_month": fname})
        )
        grp["date_block_num"] = grp["date_block_num"] + 1
        matrix = matrix.merge(grp, on=mc, how="left")

    lag_cols = list(dict.fromkeys(
        [c for c in matrix.columns if "lag_" in c]
        + [
            "shop_mean_lag_1",
            "item_mean_lag_1",
            "cat_mean_lag_1",
            "cat_lvl1_mean_lag_1",
            "cat_lvl2_mean_lag_1",
        ]
    ))
    matrix[lag_cols] = matrix[lag_cols].fillna(0).astype(np.float32)

    return matrix


# ── Métricas ──────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=== load_metrics.py ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        print("1) Descargando artefactos de S3 …")
        download_s3(tmp)

        print("2) Construyendo matriz de features …")
        matrix = build_matrix(tmp)

        print("3) Cargando modelo …")
        model = joblib.load(tmp / "model.joblib")

        # ── Mes de validación (33) ────────────────────────────────────────────
        valid = matrix[matrix["date_block_num"] == VALID_MONTH].copy()
        valid["pred_model"] = np.clip(model.predict(valid[FEATURE_COLS]), CLIP_MIN, CLIP_MAX)
        # Naive forecast = ventas del mes anterior (lag_1) = date_block_num 32
        valid["pred_naive"] = valid["item_cnt_month_lag_1"].fillna(0).clip(CLIP_MIN, CLIP_MAX)
        y_true = valid["item_cnt_month"].values

        print(f"   Filas en validación: {len(valid):,}")

        # ── Calcular métricas por grupo ───────────────────────────────────────
        metrics_rows = []

        # 1) Global
        metrics_rows.append({
            "group_key": "all",
            "rmse": rmse(y_true, valid["pred_model"]),
            "mae":  mae(y_true, valid["pred_model"]),
            "naive_rmse": rmse(y_true, valid["pred_naive"]),
        })
        print(f"   RMSE global  modelo: {metrics_rows[-1]['rmse']:.4f}  naive: {metrics_rows[-1]['naive_rmse']:.4f}")

        # 2) Por categoría
        for cat_id, grp in valid.groupby("item_category_id"):
            yt = grp["item_cnt_month"].values
            metrics_rows.append({
                "group_key": f"category:{int(cat_id)}",
                "rmse": rmse(yt, grp["pred_model"]),
                "mae":  mae(yt, grp["pred_model"]),
                "naive_rmse": rmse(yt, grp["pred_naive"]),
            })

        # 3) Por tienda
        for shop_id, grp in valid.groupby("shop_id"):
            yt = grp["item_cnt_month"].values
            metrics_rows.append({
                "group_key": f"shop:{int(shop_id)}",
                "rmse": rmse(yt, grp["pred_model"]),
                "mae":  mae(yt, grp["pred_model"]),
                "naive_rmse": rmse(yt, grp["pred_naive"]),
            })

        print(f"   Métricas calculadas: {len(metrics_rows)} grupos")

        # ── Insertar en RDS ───────────────────────────────────────────────────
        print("4) Conectando a RDS …")
        conn = get_connection()
        try:
            cur = conn.cursor()

            # Limpiar métricas previas para evitar duplicados
            cur.execute("DELETE FROM evaluation_metrics")

            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO evaluation_metrics
                    (group_key, rmse, mae, naive_rmse)
                VALUES %s
                """,
                [(r["group_key"], r["rmse"], r["mae"], r["naive_rmse"]) for r in metrics_rows],
            )

            conn.commit()
            print(f"✓ evaluation_metrics: {len(metrics_rows)} filas insertadas")
        finally:
            conn.close()

    print("=== done ===")


if __name__ == "__main__":
    main()
