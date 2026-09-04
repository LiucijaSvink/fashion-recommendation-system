"""Stateless preprocessing helpers for transactions. Column names come from
`Schema` (defaults to H&M)."""
from __future__ import annotations

from datetime import date

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from ..schema import DEFAULT_SCHEMA, Schema


def dedup_transactions(
    transactions: DataFrame,
    schema: Schema = DEFAULT_SCHEMA,
    extra_first_cols: list[str] | None = None,
) -> DataFrame:
    """Collapse repeated (user, item, day) rows to one. For each name in
    `extra_first_cols`, keep its first non-null value per group. The output
    DataFrame contains only the join keys (`schema.user`, `schema.item`,
    `schema.date`) plus `extra_first_cols` — any other columns in `transactions` are dropped."""
    aggs = [F.first(c, ignorenulls=True).alias(c) for c in (extra_first_cols or [])]
    if not aggs:
        deduped = transactions.select(schema.user, schema.item, schema.date).distinct()
        return deduped
    deduped = transactions.groupBy(schema.user, schema.item, schema.date).agg(*aggs)
    return deduped


def filter_date_range(
    transactions: DataFrame,
    schema: Schema = DEFAULT_SCHEMA,
    start: date | str | None = None,
    end: date | str | None = None,
) -> DataFrame:
    """Keep only rows where `schema.date` lies in [`start`, `end`] (inclusive on
    both ends). Either bound can be None to leave that side open."""
    if start is not None:
        transactions = transactions.filter(F.col(schema.date) >= F.lit(start))
    if end is not None:
        transactions = transactions.filter(F.col(schema.date) <= F.lit(end))
    return transactions


def filter_min_purchases(
    transactions: DataFrame,
    min_n: int = 1,
    schema: Schema = DEFAULT_SCHEMA,
) -> DataFrame:
    """Drop users with fewer than `min_n` transactions."""
    keep = (transactions.groupBy(schema.user).agg(F.count("*").alias("_n"))
            .filter(F.col("_n") >= min_n).select(schema.user))
    # no count() to pick a join hint: that runs the aggregation twice (once to decide,
    # once to join) and adaptive execution already switches to a broadcast when the
    # build side turns out to be small
    return transactions.join(keep, on=schema.user, how="inner")


def add_week(
    transactions: DataFrame,
    schema: Schema = DEFAULT_SCHEMA,
    col: str = "week",
) -> DataFrame:
    """Add an integer week index (weeks since the earliest date)."""
    first = transactions.select(F.min(schema.date)).collect()[0][0]
    with_week = transactions.withColumn(col, (F.datediff(F.col(schema.date), F.lit(first)) / 7).cast("int"))
    return with_week
