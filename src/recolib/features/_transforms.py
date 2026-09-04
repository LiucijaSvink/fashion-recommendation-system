"""Transforms that aren't sources: source-flag derivation, label join, post-join ratios.

These don't fit the `FeatureSource` shape — they either operate on the candidate
dataframe directly (no separate block to join in) or carry no configuration to
hold as state. Plain functions are the right tool.
"""
from __future__ import annotations

from typing import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..schema import Schema


def source_flags(df: DataFrame, sources: Sequence[str]) -> DataFrame:
    """Add `src_<source>` flags + `n_sources` count from per-source rank columns.

    A null `<source>_rank` means the candidate wasn't surfaced by that source.
    The flags and total give the reranker direct access to retrieval-stage
    coverage signal.
    """
    out = df
    for src in sources:
        out = out.withColumn(f"src_{src}", F.col(f"{src}_rank").isNotNull().cast("int"))
    out = out.withColumn("n_sources", sum(F.col(f"src_{s}") for s in sources))
    return out


def attach_labels(df: DataFrame, label_df: DataFrame | None, schema: Schema) -> DataFrame:
    """Join positive `(user, item)` pairs from `label_df` → `label` column (1/0).

    `label_df` is the held-out future-week purchases (val for training folds).
    For the submission fold pass `None` and every row gets `label = 0`.
    """
    if label_df is None:
        unlabeled = df.withColumn("label", F.lit(0))
        return unlabeled
    s = schema
    pos = label_df.select(s.user, s.item).distinct().withColumn("label", F.lit(1))
    labeled = (
        df.join(pos, on=[s.user, s.item], how="left")
          .withColumn("label", F.coalesce(F.col("label"), F.lit(0)))
    )
    return labeled


def derived_ratios(df: DataFrame) -> DataFrame:
    """Post-join cross features computed from already-joined source columns.

    Requires the columns produced by `UserAggregates`, `ItemAggregates`,
    `UserItemHistory`, `ItemBuyerDemographic`, `CustomerMetadata`, and
    `CategoryAffinity(key="department_no")` to be present.
    """
    enriched = (
        df
        .withColumn("price_ratio",
                    F.col("item_avg_price") / (F.col("user_avg_price") + 1e-6))
        .withColumn("user_dept_ratio",
                    F.col("user_dept_purchases") / (F.col("user_purchases") + 1e-6))
        .withColumn("bought_before", (F.col("user_item_purchases") > 0).cast("int"))
        .withColumn("age_gap", F.col("age") - F.col("item_mean_buyer_age"))
        .withColumn("user_item_ratio",
                    F.col("user_item_purchases") / (F.col("item_purchases_all") + F.lit(1.0)))
    )
    return enriched
