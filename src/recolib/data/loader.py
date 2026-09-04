"""DataLoader — generic, format-agnostic reader into Spark DataFrames.

`load()` reads csv/parquet/json and returns a DataFrame. `apply_casts()` is a
separate stateless transform you apply to that DataFrame. Compose:

    loader = DataLoader(spark, base_dir="/data")
    transactions = DataLoader.apply_casts(loader.load("transactions.csv"),
                                {"t_dat": "date", "price": "double"})
"""
from __future__ import annotations

import os

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession


class DataLoader:
    def __init__(self, spark: SparkSession, base_dir: str | None = None):
        self.spark = spark
        self.base_dir = base_dir

    def _resolve(self, source: str) -> str:
        if self.base_dir and not os.path.isabs(source):
            return os.path.join(self.base_dir, source)
        return source

    @staticmethod
    def _infer_fmt(path: str) -> str:
        for ext in ("csv", "json", "parquet"):
            if path.rstrip("/").endswith("." + ext):
                return ext
        return "parquet"  # Spark parquet dirs have no extension

    def load(
        self,
        source: str,
        fmt: str | None = None,
        header: bool = True,
        **read_options,
    ) -> DataFrame:
        """Read csv/parquet/json into a Spark DataFrame. No transformations."""
        path = self._resolve(source)
        fmt = fmt or self._infer_fmt(path)
        reader = self.spark.read.options(**read_options)
        if fmt == "csv":
            df = reader.option("header", header).csv(path)
        elif fmt == "json":
            df = reader.json(path)
        elif fmt == "parquet":
            df = reader.parquet(path)
        else:
            raise ValueError(f"unsupported fmt: {fmt!r}")
        return df

    @staticmethod
    def apply_casts(
        df: DataFrame,
        casts: dict[str, str],
        date_format: str = "yyyy-MM-dd",
    ) -> DataFrame:
        """Apply {column -> type} casts. `type` is a Spark DDL type (e.g. "double",
        "int") or the special "date" (parsed with `date_format`)."""
        for col, typ in (casts or {}).items():
            if typ == "date":
                df = df.withColumn(col, F.to_date(col, date_format))
            else:
                df = df.withColumn(col, F.col(col).cast(typ))
        return df
