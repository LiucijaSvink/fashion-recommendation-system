"""Stage 5 — score candidates and measure ranking quality.

Scoring runs inside Spark: the fitted ranker is broadcast to the executors, each
partition predicts its own rows, and only the top-k per customer comes back to the
driver. A fold is therefore read once, and the driver receives ~12 rows per customer
rather than the couple of hundred candidates each one has.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructField, StructType

from ..cache import cached
from ..config import PipelineConfig
from ..metrics import hit_rate_at_k, mapk, ndcg_at_k, precision_at_k, recall_at_k
from ..modeling import LambdaRanker

log = logging.getLogger(__name__)


def sample_customers(features: DataFrame, cfg: PipelineConfig, n_users: int) -> DataFrame:
    """A deterministic customer sample — same config, same customers, every run."""
    user = cfg.schema.user
    everyone = features.select(user).distinct()
    total = everyone.count()
    if n_users >= total:
        return everyone
    fraction = min(1.0, 1.5 * n_users / total)   # oversample, then take exactly n
    return (everyone.sample(withReplacement=False, fraction=fraction, seed=cfg.seed)
            .limit(n_users))



def evaluation_users(features: DataFrame, cfg: PipelineConfig) -> DataFrame | None:
    """The customers to score: a fixed sample, or `None` meaning the whole population.

    `score()` reads `None` as "walk the table in chunks", so this is the one place that
    decides between the two and both the notebook and `evaluate()` go through it.
    """
    return sample_customers(features, cfg, cfg.eval_sample_users) if cfg.eval_sample_users else None


def buyers_of(recommendations: dict[str, list], truth: dict[str, set]) -> list:
    """Scored customers who actually bought in the held-out week."""
    return [user for user in recommendations if truth.get(user)]


def score(
    model: LambdaRanker,
    features: DataFrame,
    cfg: PipelineConfig,
    *,
    users: DataFrame | None = None,
    backfill: dict | None = None,
) -> dict[str, list]:
    """`{customer: [top-k items]}`, predicted on the executors.

    The ranker is broadcast once and applied partition-wise, so the feature table is
    read a single time and never materialised on the driver. Ties in the model score
    are broken by item id, which makes the output reproducible run to run.
    """
    user, item = cfg.schema.user, cfg.schema.item
    if users is not None:
        features = features.join(users.select(user).distinct(), on=user)

    feature_columns = list(model.feature_cols or [])
    if not feature_columns:
        raise ValueError("model has no feature_cols — fit it before scoring")
    prediction_schema = StructType([
        features.schema[user], features.schema[item], StructField("score", DoubleType()),
    ])
    broadcast_model = features.sparkSession.sparkContext.broadcast(model)

    def predict(batches):
        ranker = broadcast_model.value
        for batch in batches:
            yield pd.DataFrame({user: batch[user], item: batch[item],
                                "score": ranker.score(batch)})

    scored = (features.select(user, item, *feature_columns)
                      .mapInPandas(predict, schema=prediction_schema))

    best_first = Window.partitionBy(user).orderBy(F.col("score").desc(), F.col(item))
    top_k = (scored.withColumn("rank", F.row_number().over(best_first))
                   .filter(F.col("rank") <= cfg.eval_k))
    per_customer = (
        top_k.groupBy(user)
             .agg(F.sort_array(F.collect_list(F.struct("rank", item))).alias("ranked"))
             .select(user, F.col(f"ranked.{item}").alias("items"))
             .toPandas()
    )
    log.info("scored %s customers on %s", f"{len(per_customer):,}", cfg.eval_k)

    recommendations = {row[0]: list(row[1]) for row in per_customer.itertuples(index=False)}
    if backfill:
        recommendations = _apply_backfill(recommendations, backfill, cfg.eval_k)
    return recommendations


def _apply_backfill(recommendations: dict[str, list], backfill: dict, k: int) -> dict[str, list]:
    """Pad short lists and add customers who had no candidates at all."""
    for customer, fallback in backfill.items():
        current = recommendations.get(customer, [])
        if len(current) < k:
            seen = set(current)
            recommendations[customer] = (current + [i for i in fallback if i not in seen])[:k]
    return recommendations


def ground_truth(val: DataFrame, cfg: PipelineConfig) -> dict[str, set]:
    """`{customer: {items bought in the held-out week}}`."""
    pdf = val.select(cfg.schema.user, cfg.schema.item).distinct().toPandas()
    return pdf.groupby(cfg.schema.user)[cfg.schema.item].apply(set).to_dict()


def evaluate(
    spark: SparkSession, cfg: PipelineConfig, model: LambdaRanker, fold: str
) -> dict[str, float]:
    """MAP@k / recall@k for one fold, over a customer sample or the full population."""
    features = spark.read.parquet(cfg.path(fold, "features"))
    val = spark.read.parquet(cfg.path(fold, "val"))
    truth = ground_truth(val, cfg)

    recommendations = score(model, features, cfg, users=evaluation_users(features, cfg))

    scored_users = list(recommendations)
    buyers = buyers_of(recommendations, truth)
    result = {
        "customers scored": len(scored_users),
        "buyers": len(buyers),
        # the full accuracy set, so a per-fold table carries the same columns as 6.2
        **accuracy_metrics(recommendations, truth, cfg, scored_users),
        f"MAP@{cfg.eval_k} (buyers)": mapk(recommendations, truth, k=cfg.eval_k, users=buyers),
    }
    log.info("%s: %s", fold, result)
    return result


def popular_items(train: DataFrame, cfg: PipelineConfig, train_end: date,
                  k: int) -> list[str]:
    """Top-k bestsellers of the recent window — the non-personalised reference."""
    cutoff = train_end - timedelta(weeks=cfg.popularity_weeks)
    top = (train.filter(F.col(cfg.schema.date) > F.lit(cutoff))
                .groupBy(cfg.schema.item).agg(F.count("*").alias("n"))
                .orderBy(F.col("n").desc()).limit(k))
    return [row[cfg.schema.item] for row in top.collect()]


def accuracy_metrics(
    recommendations: dict[str, list], truth: dict[str, set], cfg: PipelineConfig,
    users: list | None = None,
) -> dict[str, float]:
    """The four accuracy numbers, over one fixed population."""
    population = users if users is not None else list(recommendations)
    k = cfg.eval_k
    return {
        f"MAP@{k}": mapk(recommendations, truth, k=k, users=population),
        f"NDCG@{k}": ndcg_at_k(recommendations, truth, k=k, users=population),
        f"hit-rate@{k}": hit_rate_at_k(recommendations, truth, k=k, users=population),
        f"precision@{k}": precision_at_k(recommendations, truth, k=k, users=population),
        f"recall@{k}": recall_at_k(recommendations, truth, k=k, users=population),
    }


def compare_to_baseline(
    recommendations: dict[str, list], truth: dict[str, set], cfg: PipelineConfig,
    baseline_items: list[str], users: list | None = None,
) -> pd.DataFrame:
    """Model against 'recommend the same bestsellers to everyone'.

    Without this row every accuracy number is unanchored — the baseline is what the
    business would do with no model at all.
    """
    population = users if users is not None else list(recommendations)
    baseline = {user: list(baseline_items) for user in population}
    table = pd.DataFrame({
        "popularity baseline": accuracy_metrics(baseline, truth, cfg, population),
        "model": accuracy_metrics(recommendations, truth, cfg, population),
    })
    table["lift"] = table["model"] / table["popularity baseline"].replace(0, float("nan"))
    return table


def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Several evaluations (e.g. one per fold) as one table."""
    return pd.DataFrame(results).T

@dataclass(frozen=True)
class ScoredFold:
    """One fold, scored: the recommendations and everything needed to judge them.

    Bundling these keeps callers from re-deriving the held-out truth or the baseline
    (and from scoring the same fold twice, which at full population is an hour).
    """
    fold: str
    recommendations: dict
    truth: dict
    baseline: list

    @property
    def customers(self) -> list:
        return list(self.recommendations)

    @property
    def buyers(self) -> list:
        return buyers_of(self.recommendations, self.truth)

    def __str__(self) -> str:
        return (f"{self.fold}: {len(self.customers):,} customers scored, "
                f"{len(self.buyers):,} bought that week")


def baseline_items(spark: SparkSession, cfg: PipelineConfig, fold: str) -> list:
    """The non-personalised reference: the fold's recent bestsellers."""
    train = spark.read.parquet(cfg.path(fold, "train"))
    train_end = train.select(F.max(cfg.schema.date)).collect()[0][0]
    return popular_items(train, cfg, train_end, cfg.eval_k)


def score_fold(
    spark: SparkSession, cfg: PipelineConfig, model: LambdaRanker, fold: str
) -> ScoredFold:
    """Score one fold once, and carry the results around rather than recomputing them.

    Cached on disk: at full population this is the most expensive single step of a run,
    and every table in the results section is derived from it.
    """
    def run() -> ScoredFold:
        features = spark.read.parquet(cfg.path(fold, "features"))
        val = spark.read.parquet(cfg.path(fold, "val"))
        return ScoredFold(
            fold=fold,
            recommendations=score(model, features, cfg, users=evaluation_users(features, cfg)),
            truth=ground_truth(val, cfg),
            baseline=baseline_items(spark, cfg, fold),
        )

    return cached(cfg, f"score_fold.{fold}", run)
