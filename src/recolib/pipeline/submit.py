"""Stage 6 — write the Kaggle submission.

Every customer in `sample_submission.csv` must appear exactly once with 12 items, so
customers with too few scored candidates are padded with last-week bestsellers.
"""
from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PipelineConfig
from ..modeling import LambdaRanker
from .evaluation import popular_items, score
from .io import load_fold

log = logging.getLogger(__name__)


def write_submission(
    spark: SparkSession, cfg: PipelineConfig, model: LambdaRanker, *, fold: str | None = None
) -> pd.DataFrame:
    """Score the submission fold, pad short lists, write and validate the CSV."""
    fold = fold or cfg.submission_fold
    inputs = load_fold(spark, cfg, fold)
    features = spark.read.parquet(cfg.path(fold, "features"))

    fallback = popular_items(inputs.train, cfg, inputs.train_end, cfg.eval_k)
    all_customers = [row[cfg.schema.user] for row in
                     inputs.customers.select(cfg.schema.user).distinct().collect()]
    backfill = {customer: fallback for customer in all_customers}

    recommendations = score(model, features, cfg.with_(eval_sample_users=None), backfill=backfill)
    submission = pd.DataFrame(
        [{"customer_id": customer, "prediction": " ".join(items)}
         for customer, items in recommendations.items()]
    )
    submission.to_csv(cfg.submission_path, index=False)
    log.info("wrote %s rows -> %s", f"{len(submission):,}", cfg.submission_path)

    validate(submission, cfg)
    return submission


def validate(submission: pd.DataFrame, cfg: PipelineConfig) -> None:
    """Fail loudly if the file would be rejected by Kaggle."""
    sample = pd.read_csv(cfg.sample_submission_csv, usecols=["customer_id"])
    expected, got = set(sample["customer_id"]), set(submission["customer_id"])
    if missing := expected - got:
        raise ValueError(f"submission missing {len(missing):,} customers")
    if extra := got - expected:
        raise ValueError(f"submission has {len(extra):,} unexpected customers")
    lengths = submission["prediction"].str.split().str.len()
    if (bad := int((lengths != cfg.eval_k).sum())):
        raise ValueError(f"{bad:,} rows do not have exactly {cfg.eval_k} predictions")
    log.info("submission validated against %s", cfg.sample_submission_csv)
