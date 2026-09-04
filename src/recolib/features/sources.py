"""Concrete feature sources.

Each returns `(block_df, join_keys)` from `compute(context)`. Column names
come from `context.schema` for the user/item/date keys; everything else is
the feature column names produced (declared in `self.produces`).

The feature set, broken into 8 reusable sources:
  * User-level aggregates (active/inactive coalesce):  `UserAggregates`
  * Item-level multi-window aggregates:                `ItemAggregates`
  * User × category affinity (parameterized by key):   `CategoryAffinity`
  * User × item lifetime repurchase:                    `UserItemHistory`
  * Item demographic (mean buyer age):                  `ItemBuyerDemographic`
  * Customer metadata (age + indexed categoricals):     `CustomerMetadata`
  * Article catalog codes:                              `ArticleMetadata`
  * Co-purchase score from itemcf neighbors:            `ItemCFScore`
"""
from __future__ import annotations

from datetime import timedelta

from pyspark.ml.feature import StringIndexer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base import FeatureContext, FeatureSource


class UserAggregates(FeatureSource):
    """User behavioral features with active/inactive coalesce.

    Active users (any purchase in the last `recent_weeks`) get aggregates over
    the recent window. Inactive users get the SAME columns over their full
    history (their only signal). Unioned + an `is_active` flag.

    Only inactive users pay the lifetime-aggregation cost.
    """
    name = "user_aggregates"
    produces = [
        "user_purchases", "user_n_unique_items",
        "user_days_since_first", "user_days_since_last",
        "user_avg_price", "user_std_price", "user_max_price",
        "user_avg_channel", "is_active",
    ]

    def __init__(self, recent_weeks: int = 12):
        self.recent_weeks = recent_weeks

    def _agg(self, d, s, end):
        block = d.groupBy(s.user).agg(
            F.count("*").alias("user_purchases"),
            F.approx_count_distinct(s.item).alias("user_n_unique_items"),
            F.datediff(F.lit(end), F.min(s.date)).alias("user_days_since_first"),
            F.datediff(F.lit(end), F.max(s.date)).alias("user_days_since_last"),
            F.avg("price").alias("user_avg_price"),
            F.stddev("price").alias("user_std_price"),
            F.max("price").alias("user_max_price"),
            F.avg(F.col("sales_channel_id").cast("double")).alias("user_avg_channel"),
        )
        return block

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        s, end = context.schema, context.end
        history = context.history
        recent = history.filter(F.col(s.date) >= F.lit(end - timedelta(weeks=self.recent_weeks)))
        active = recent.select(s.user).distinct()
        inactive_hist = history.join(active, on=s.user, how="leftanti")

        block = (
            self._agg(recent, s, end).withColumn("is_active", F.lit(1))
            .unionByName(self._agg(inactive_hist, s, end).withColumn("is_active", F.lit(0)))
        )
        return block, [s.user]


class ItemAggregates(FeatureSource):
    """Item popularity / recency / price over multiple windows + lifetime.

    Produces one `item_purchases_{w}w` column per window plus an `_all` total
    and a `item_repurchase_rate = total / unique_buyers`.
    """
    name = "item_aggregates"

    def __init__(self, windows: tuple[int, ...] = (1, 4, 12)):
        self.windows = tuple(windows)
        self.produces = (
            [f"item_purchases_{w}w" for w in self.windows]
            + [
                "item_purchases_all", "item_n_unique_buyers",
                "item_days_since_first_sale", "item_days_since_last_sale",
                "item_avg_price", "item_max_price", "item_repurchase_rate",
            ]
        )

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        s, end = context.schema, context.end
        windowed = [
            F.sum(F.when(F.col(s.date) >= F.lit(end - timedelta(weeks=w)), 1).otherwise(0))
             .alias(f"item_purchases_{w}w")
            for w in self.windows
        ]
        block = (
            context.transactions.groupBy(s.item).agg(
                *windowed,
                F.count("*").alias("item_purchases_all"),
                F.countDistinct(s.user).alias("item_n_unique_buyers"),
                F.datediff(F.lit(end), F.min(s.date)).alias("item_days_since_first_sale"),
                F.datediff(F.lit(end), F.max(s.date)).alias("item_days_since_last_sale"),
                F.avg("price").alias("item_avg_price"),
                F.max("price").alias("item_max_price"),
            )
            .withColumn("item_repurchase_rate",
                        F.col("item_purchases_all") / F.col("item_n_unique_buyers"))
        )
        return block, [s.item]


class CategoryAffinity(FeatureSource):
    """User × article-category purchase count, with active/inactive coalesce.

    `key` is a column on `context.articles` (e.g. `product_code`,
    `colour_group_code`, `section_no`). Produces `user_{out_name or key}_purchases`.
    To add a new category-affinity feature in production, instantiate this
    class with the article column name — no new code required.
    """

    def __init__(self, key: str, recent_weeks: int = 12, out_name: str | None = None):
        self.key = key
        self.recent_weeks = recent_weeks
        suffix = out_name or key
        self.name = f"category_affinity_{suffix}"
        self._out_col = f"user_{suffix}_purchases"
        self.produces = [self._out_col]

    def _cat_count(self, d, articles, s):
        counts = (
            d.join(F.broadcast(articles), on=s.item, how="left")
             .groupBy(s.user, self.key)
             .agg(F.count("*").alias(self._out_col))
        )
        return counts

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        s, end = context.schema, context.end
        history = context.history
        articles = context.articles.select(s.item, self.key)
        recent = history.filter(F.col(s.date) >= F.lit(end - timedelta(weeks=self.recent_weeks)))
        active = recent.select(s.user).distinct()
        inactive_hist = history.join(active, on=s.user, how="leftanti")

        block = (self._cat_count(recent, articles, s)
                 .unionByName(self._cat_count(inactive_hist, articles, s)))
        return block, [s.user, self.key]


class UserItemHistory(FeatureSource):
    """Lifetime user × item repurchase signal (count + recency).

    Uses full history for ALL users — matches the full-history repurchase
    candidates so active users still get credit for old favourites.
    """
    name = "user_item_history"
    produces = ["user_item_purchases", "user_item_days_since_last"]

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        s, end = context.schema, context.end
        block = context.history.groupBy(s.user, s.item).agg(
            F.count("*").alias("user_item_purchases"),
            F.datediff(F.lit(end), F.max(s.date)).alias("user_item_days_since_last"),
        )
        return block, [s.user, s.item]


class ItemBuyerDemographic(FeatureSource):
    """Mean age of an item's historical buyers (full history)."""
    name = "item_buyer_demographic"
    produces = ["item_mean_buyer_age"]

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        s = context.schema
        cust_age = context.customers.select(
            s.user, F.col("age").cast("double").alias("_buyer_age"),
        )
        block = (
            context.history.join(cust_age, on=s.user, how="inner")
            .groupBy(s.item).agg(F.avg("_buyer_age").alias("item_mean_buyer_age"))
        )
        return block, [s.item]


class CustomerMetadata(FeatureSource):
    """Customer demographics: numeric age + string-indexed categoricals.

    The default categorical columns are `club_member_status` and
    `fashion_news_frequency`. Each gets a `<col>_idx` numeric encoding.
    """
    name = "customer_metadata"

    def __init__(
        self,
        categorical_cols: tuple[str, ...] = ("club_member_status", "fashion_news_frequency"),
    ):
        self.categorical_cols = list(categorical_cols)
        self.produces = ["age"] + [f"{c}_idx" for c in self.categorical_cols]

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        s = context.schema
        block = context.customers.select(
            s.user,
            F.col("age").cast("double").alias("age"),
            *self.categorical_cols,
        )
        for c in self.categorical_cols:
            block = (
                StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                .fit(block).transform(block).drop(c)
            )
        return block, [s.user]


class ArticleMetadata(FeatureSource):
    """Article catalog categorical codes (cast to double for LightGBM).

    LightGBM treats these as ordinals / categoricals depending on training
    config. Defaults are the six numeric article codes.
    """
    name = "article_metadata"

    def __init__(
        self,
        codes: tuple[str, ...] = (
            "garment_group_no", "section_no", "index_group_no",
            "colour_group_code", "product_type_no", "department_no",
        ),
    ):
        self.codes = list(codes)
        self.produces = list(codes)

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        s = context.schema
        block = context.articles.select(
            s.item, *[F.col(c).cast("double").alias(c) for c in self.codes],
        )
        return block, [s.item]


class ItemCFScore(FeatureSource):
    """Co-purchase score with the user's recent seeds, from itemcf neighbors.

    Requires `context.neighbors` — the same neighbor matrix used by the
    `ItemCF` candidate source (typically checkpointed during retrieval and
    reloaded for features).
    """
    name = "itemcf_score"
    produces = ["itemcf_score", "itemcf_max_score"]

    def __init__(self, seed_weeks: int = 4):
        self.seed_weeks = seed_weeks

    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        if context.neighbors is None:
            raise ValueError(
                "ItemCFScore requires context.neighbors (the itemcf neighbor table). "
                "Pass it via FeatureContext or remove ItemCFScore from the source list."
            )
        s, end = context.schema, context.end
        seeds = (
            context.transactions
            .filter(F.col(s.date) >= F.lit(end - timedelta(weeks=self.seed_weeks)))
            .select(s.user, F.col(s.item).alias("seed")).distinct()
        )
        block = (
            seeds.join(context.neighbors, on="seed")
            .groupBy(s.user, s.item).agg(
                F.sum("cocount").alias("itemcf_score"),
                F.max("cocount").alias("itemcf_max_score"),
            )
        )
        return block, [s.user, s.item]
