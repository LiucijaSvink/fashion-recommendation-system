"""Stage 4 — train the LambdaRank reranker.

Positives are every retrieved candidate the customer actually bought; negatives are
downsampled to `negatives_per_positive`. With more than one training fold the ranking
group is (fold, customer), so the same customer in two folds is two ranking problems.
"""
from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PipelineConfig
from ..modeling import LambdaRanker

log = logging.getLogger(__name__)

FOLD_COLUMN = "_fold"


def _downsample(features: DataFrame, cfg: PipelineConfig) -> tuple[DataFrame, int, int]:
    """Keep all positives, sample negatives down to the configured ratio."""
    positives = features.filter(F.col("label") == 1)
    negatives = features.filter(F.col("label") == 0)
    n_pos, n_neg = positives.count(), negatives.count()
    if n_pos == 0:
        raise ValueError("no positive labels — was this the submission fold?")
    fraction = min(1.0, cfg.negatives_per_positive * n_pos / n_neg) if n_neg else 0.0
    sampled = negatives.sample(withReplacement=False, fraction=fraction, seed=cfg.seed)
    return positives.unionByName(sampled), n_pos, n_neg


def build_training_frame(spark: SparkSession, cfg: PipelineConfig) -> pd.DataFrame:
    """Downsampled training rows across `cfg.train_folds`, collected to pandas."""
    parts = []
    for index, fold in enumerate(cfg.train_folds):
        features = spark.read.parquet(cfg.path(fold, "features"))
        sampled, n_pos, n_neg = _downsample(features, cfg)
        log.info("%s: %s positives, %s negatives -> %s:1", fold, f"{n_pos:,}", f"{n_neg:,}",
                 cfg.negatives_per_positive)
        parts.append(sampled.withColumn(FOLD_COLUMN, F.lit(index)))

    frame = parts[0]
    for part in parts[1:]:
        frame = frame.unionByName(part)
    return frame.toPandas()


def train(spark: SparkSession, cfg: PipelineConfig) -> tuple[LambdaRanker, pd.DataFrame]:
    """Fit the reranker. Returns the model and the frame it was trained on."""
    train_pdf = build_training_frame(spark, cfg)
    group_cols = ([FOLD_COLUMN, cfg.schema.user] if len(cfg.train_folds) > 1
                  else [cfg.schema.user])
    model = LambdaRanker(schema=cfg.schema, **cfg.lgbm_params).fit(
        train_pdf, group_cols=group_cols
    )
    log.info("trained on %s rows, grouped by %s", f"{len(train_pdf):,}", group_cols)
    return model, train_pdf
