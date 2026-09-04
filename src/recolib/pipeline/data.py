"""Stage 1 — raw CSV to fold parquets.

Writes, under `cfg.out_dir`:
  transactions/            full cast history, the source of lifetime signals
  articles/ customers/     catalog tables, so later stages never re-read raw CSV
  {fold}/train, {fold}/val one directory per rolling fold, plus a submission fold
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PipelineConfig
from ..data import DataLoader, TimeSeriesSplitter
from ..data.splitter import Fold

log = logging.getLogger(__name__)

CASTS = {"t_dat": "date", "price": "double", "sales_channel_id": "int"}


def _write_once(df: DataFrame, path: str, *, overwrite: bool) -> None:
    """Write `df` to `path`, skipping the work when it is already there."""
    mode = "overwrite" if overwrite else "ignore"
    if not overwrite and _exists(df.sparkSession, path):
        # "ignore" is silent by design, which is how a stale artifact from an older
        # build survives a re-run and quietly feeds every number downstream
        log.warning("%s already exists — keeping it; pass overwrite=True to rebuild", path)
    df.write.mode(mode).parquet(path)


def _exists(spark: SparkSession, path: str) -> bool:
    jvm, hadoop = spark._jvm, spark._jsc.hadoopConfiguration()
    jpath = jvm.org.apache.hadoop.fs.Path(path)
    return jpath.getFileSystem(hadoop).exists(jpath)


def load_transactions(spark: SparkSession, cfg: PipelineConfig) -> DataFrame:
    """Raw transactions CSV with types cast per `CASTS`."""
    loader = DataLoader(spark)
    return DataLoader.apply_casts(loader.load(cfg.transactions_csv), casts=CASTS)


def build_folds(transactions: DataFrame, cfg: PipelineConfig) -> list[Fold]:
    """Rolling time-series folds + a submission fold, per config."""
    splitter = TimeSeriesSplitter(
        window_end=cfg.window_end,
        train_weeks=cfg.train_weeks,
        val_weeks=cfg.val_weeks,
        n_folds=cfg.n_folds,
        with_submission=True,
        schema=cfg.schema,
    )
    return splitter.split(transactions)


def fold_name(cfg: PipelineConfig, fold: Fold, index: int) -> str:
    """Directory name for a fold under `cfg.out_dir`.

    The splitter names folds `fold_0...` regardless of configuration; the on-disk
    layout follows `cfg.fold_prefix`, so the two are mapped here rather than assumed
    to match.
    """
    return cfg.submission_fold if fold.is_submission else f"{cfg.fold_prefix}_{index}"


def prepare(spark: SparkSession, cfg: PipelineConfig, *, overwrite: bool = False) -> list[Fold]:
    """Run stage 1. Returns the folds so callers can inspect their boundaries."""
    transactions = load_transactions(spark, cfg)
    _write_once(transactions, cfg.history_path, overwrite=overwrite)

    for csv_path, out_path in ((cfg.articles_csv, cfg.articles_path),
                               (cfg.customers_csv, cfg.customers_path)):
        _write_once(spark.read.option("header", True).csv(csv_path), out_path, overwrite=overwrite)

    folds = build_folds(transactions, cfg)
    for index, fold in enumerate(folds):
        name = fold_name(cfg, fold, index)
        _write_once(fold.train, cfg.path(name, "train"), overwrite=overwrite)
        # the submission fold has no future week; an empty val keeps the layout uniform
        val = fold.val if fold.val is not None else fold.train.limit(0)
        _write_once(val, cfg.path(name, "val"), overwrite=overwrite)
        log.info("prepared %s: train %s -> %s", name,
                 fold.meta["train_start"], fold.meta["train_end"])
    return folds


def fold_table(spark: SparkSession, cfg: PipelineConfig, folds: list[Fold]) -> pd.DataFrame:
    """Fold boundaries + row counts + an explicit leakage check, as a pandas table."""
    import pandas as pd

    rows = []
    for index, fold in enumerate(folds):
        name = fold_name(cfg, fold, index)
        train = spark.read.parquet(cfg.path(name, "train"))
        val = spark.read.parquet(cfg.path(name, "val"))
        n_val = val.count()
        if n_val:
            train_max = train.select(F.max(cfg.schema.date)).collect()[0][0]
            val_min = val.select(F.min(cfg.schema.date)).collect()[0][0]
            leak = "ok" if train_max < val_min else f"LEAK {train_max} >= {val_min}"
        else:
            leak = "n/a"
        meta = fold.meta
        rows.append({
            "fold": name,
            "train start": meta["train_start"],
            "train end": meta["train_end"],
            "val start": meta["val_start"] or "—",
            "train rows": train.count(),
            "val rows": n_val,
            "train < val": leak,
        })
    return pd.DataFrame(rows).set_index("fold")
