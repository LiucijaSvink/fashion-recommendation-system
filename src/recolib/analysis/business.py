"""Commercial reading of a set of recommendations."""
from __future__ import annotations


from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PipelineConfig


# ---------------------------------------------------------------------------
# Business-facing evaluation
#
# Accuracy metrics say how well the ranking orders items; these say what the
# recommendations would mean commercially. Caveats that belong beside every number:
# H&M's `price` is scaled, not currency, so figures are relative only; there is no
# margin or returns data, so revenue is not profit; and an offline measurement is not
# an online uplift — there is no counterfactual here.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BusinessInputs:
    """Small lookups the business panel needs, collected once."""

    item_price: dict          # item -> mean price
    new_items: set            # items launched inside the recent window
    prior_pairs: set          # (customer, item) the customer had already bought
    popular_items: list       # the global top-k
    catalogue_size: int       # articles in the catalogue, the denominator for coverage


def business_inputs(
    spark: SparkSession, cfg: PipelineConfig, fold: str, users: list, k: int | None = None,
) -> BusinessInputs:
    """Collect the lookups for `users`. Everything is restricted before collecting.

    Pass **buyers only**, not every scored customer. `prior_pairs` is the one field that
    grows with `users`, and it is read solely to split correct predictions into
    replenishment and discovery — a customer who bought nothing that week contributes no
    correct predictions, so their history is collected for nothing. On the full
    population that distinction is the difference between ~1M pairs and ~27M, which is
    what turns this from a lookup into an out-of-memory error.
    """
    user, item, date = cfg.schema.user, cfg.schema.item, cfg.schema.date
    k = k or cfg.eval_k
    train = spark.read.parquet(cfg.path(fold, "train"))
    train_end = train.select(F.max(date)).collect()[0][0]
    history = spark.read.parquet(cfg.history_path).filter(F.col(date) <= F.lit(train_end))

    prices = (history.groupBy(item).agg(F.avg("price").alias("price"))
              .toPandas().set_index(item)["price"].to_dict())

    cutoff = train_end - timedelta(weeks=cfg.seed_weeks)
    launched = history.groupBy(item).agg(F.min(date).alias("first_sale"))
    new_items = {row[item] for row in
                 launched.filter(F.col("first_sale") > F.lit(cutoff)).select(item).collect()}

    user_df = spark.createDataFrame([(u,) for u in users], [user])
    prior = (history.join(F.broadcast(user_df), on=user).select(user, item).distinct()
             .toPandas())
    prior_pairs = set(map(tuple, prior.itertuples(index=False, name=None)))

    top = (train.filter(F.col(date) > F.lit(train_end - timedelta(weeks=cfg.popularity_weeks)))
                .groupBy(item).agg(F.count("*").alias("n"))
                .orderBy(F.col("n").desc()).limit(k))
    popular = [row[item] for row in top.collect()]

    catalogue_size = spark.read.parquet(cfg.articles_path).count()

    return BusinessInputs(item_price=prices, new_items=new_items,
                          prior_pairs=prior_pairs, popular_items=popular,
                          catalogue_size=catalogue_size)


def business_panel(
    recommendations: dict, truth: dict, inputs: BusinessInputs, cfg: PipelineConfig,
) -> pd.Series:
    """Commercial reading of a set of recommendations, as metric -> value.

    revenue captured %   share of the week's spend the recommendations surfaced
    purchases captured % the same, counting purchases rather than value
    price of recommended / purchased   below 1 means it recommends cheaper than customers buy
    new arrivals in recommendations %  items launched inside the recent window
    replenishment share of hits %      correct items the customer had bought before
    catalogue coverage %               share of the catalogue that is ever recommended
    """
    price = inputs.item_price
    k = cfg.eval_k

    truth_pairs = {(u, i) for u, items in truth.items() if u in recommendations for i in items}
    hit_pairs = {(u, i) for u, items in recommendations.items()
                 for i in items[:k] if i in truth.get(u, ())}
    slots = [(u, i) for u, items in recommendations.items() for i in items[:k]]

    def total_price(pairs) -> float:
        return sum(price.get(i, 0.0) for _, i in pairs)

    revenue_captured = total_price(hit_pairs)
    revenue_available = total_price(truth_pairs)
    mean_recommended = (sum(price.get(i, 0.0) for _, i in slots) / len(slots)) if slots else 0.0
    mean_purchased = (revenue_available / len(truth_pairs)) if truth_pairs else 0.0
    replenishment = sum(1 for pair in hit_pairs if pair in inputs.prior_pairs)

    return pd.Series({
        "revenue captured %":
            100 * revenue_captured / revenue_available if revenue_available else 0.0,
        "purchases captured %":
            100 * len(hit_pairs) / len(truth_pairs) if truth_pairs else 0.0,
        "price of recommended / purchased":
            mean_recommended / mean_purchased if mean_purchased else 0.0,
        "new arrivals in recommendations %":
            100 * sum(1 for _, i in slots if i in inputs.new_items) / len(slots) if slots else 0.0,
        "replenishment share of hits %":
            100 * replenishment / len(hit_pairs) if hit_pairs else 0.0,
        "catalogue coverage %":
            100 * len({i for _, i in slots}) / inputs.catalogue_size
            if inputs.catalogue_size else 0.0,
    })


ACTIVE, LAPSED, DORMANT, COLD = (
    "active (<=4w)", "lapsed (4-12w)", "dormant (>12w)", "cold (no history)")


def activity_segments(spark: SparkSession, cfg: PipelineConfig, fold: str) -> dict:
    """Map each customer with history to active / lapsed / dormant.

    Customers absent from the result have no purchases before the cutoff; callers
    treat a missing key as `COLD` rather than being handed a dictionary the size of
    the customer base for every fold.
    """
    user, date = cfg.schema.user, cfg.schema.date
    train_end = spark.read.parquet(cfg.path(fold, "train")).select(F.max(date)).collect()[0][0]
    history = spark.read.parquet(cfg.history_path).filter(F.col(date) <= F.lit(train_end))

    days_since = F.datediff(F.lit(train_end), F.max(date))
    recency = (
        history.groupBy(user)
        .agg(F.when(days_since <= 28, ACTIVE)
              .when(days_since <= 84, LAPSED)
              .otherwise(DORMANT).alias("segment"))
        .toPandas()
    )
    return dict(zip(recency[user], recency["segment"]))


def segment_report(
    recommendations: dict, truth: dict, segments: dict, cfg: PipelineConfig,
) -> pd.DataFrame:
    """Accuracy per customer segment — where the model works and where it does not."""
    from ..pipeline.evaluation import accuracy_metrics

    rows = {}
    for label in (ACTIVE, LAPSED, DORMANT, COLD):
        members = [u for u in recommendations if segments.get(u, COLD) == label]
        if not members:
            continue
        metrics = accuracy_metrics(recommendations, truth, cfg, members)
        buyers = sum(1 for u in members if truth.get(u))
        rows[label] = {"customers": len(members), "buyers": buyers, **metrics}
    return pd.DataFrame(rows).T
