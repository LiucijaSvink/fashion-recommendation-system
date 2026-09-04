"""Loading the inputs a single fold needs.

Retrieval, feature building and submission all need the same five frames plus the
fold's cutoff date. Loading them in one place keeps the cutoff definition — "history
is everything up to and including train_end" — in exactly one place too.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PipelineConfig


@dataclass(frozen=True)
class FoldInputs:
    """Everything a stage needs about one fold."""
    name: str
    train: DataFrame          # the fold's bounded training window
    val: DataFrame | None     # the held-out week; None for the submission fold
    history: DataFrame        # ALL transactions up to train_end (lifetime signals)
    articles: DataFrame
    customers: DataFrame
    train_end: date

    @property
    def has_labels(self) -> bool:
        return self.val is not None


def load_fold(spark: SparkSession, cfg: PipelineConfig, fold: str) -> FoldInputs:
    """Read one fold's inputs. `history` is cut at the fold's train_end — no leakage."""
    train = spark.read.parquet(cfg.path(fold, "train"))
    train_end = train.select(F.max(cfg.schema.date)).collect()[0][0]
    if train_end is None:
        raise ValueError(f"{fold}: training window is empty at {cfg.path(fold, 'train')}")

    val = spark.read.parquet(cfg.path(fold, "val"))
    if val.limit(1).count() == 0:
        val = None

    history = spark.read.parquet(cfg.history_path).filter(
        F.col(cfg.schema.date) <= F.lit(train_end)
    )
    return FoldInputs(
        name=fold,
        train=train,
        val=val,
        history=history,
        articles=spark.read.parquet(cfg.articles_path),
        customers=spark.read.parquet(cfg.customers_path),
        train_end=train_end,
    )
