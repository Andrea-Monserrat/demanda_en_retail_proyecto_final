from pathlib import Path
import pandas as pd
import awswrangler as wr
import io
import boto3
import joblib
from typing import Any



def read_parquet(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if path.startswith("s3://"):
        return wr.s3.read_parquet(path)
    return pd.read_parquet(path)


def read_text(path: str) -> str:
    if path.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, key = path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read().decode("utf-8")
    return Path(path).read_text(encoding="utf-8")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    path = str(path)
    if path.startswith("s3://"):
        wr.s3.to_parquet(
            df=df,
            path=path,
            index=False,
            compression="snappy",
            mode="overwrite",
        )
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False, compression="snappy")


def write_text(text: str, path: str) -> None:
    path = str(path)

    if path.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, key = path.replace("s3://", "").split("/", 1)
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="application/json",
        )
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")

def read_csv(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        return wr.s3.read_csv(path)
    return pd.read_csv(path, encoding="utf-8", low_memory=False)

def path_exists(path: str) -> bool:
    path = str(path)

    if path.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, key = path.replace("s3://", "").split("/", 1)

        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError:
            return False

    return Path(path).exists()

def write_csv(df: pd.DataFrame, path: str) -> None:
    if path.startswith("s3://"):
        wr.s3.to_csv(df=df, path=path, index=False)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

def write_joblib(obj: Any, path: str | Path) -> None:
    path = str(path)

    if path.startswith("s3://"):
        buffer = io.BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)

        s3 = boto3.client("s3")
        bucket, key = path.replace("s3://", "").split("/", 1)
        s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)


def read_joblib(path: str | Path) -> Any:
    path = str(path)

    if path.startswith("s3://"):
        s3 = boto3.client("s3")
        bucket, key = path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return joblib.load(io.BytesIO(obj["Body"].read()))

    return joblib.load(path)