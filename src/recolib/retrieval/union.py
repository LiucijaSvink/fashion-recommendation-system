"""Combine multiple candidate sources into one table, keeping each source's rank."""
from __future__ import annotations

from functools import reduce
from typing import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .base import CandidateSource, RetrievalContext


def union_candidates(
    sources: Sequence[CandidateSource],
    transactions: DataFrame,
    context: RetrievalContext,
) -> DataFrame:
    """Generate each source and pivot per-source rank into columns.

    Returns one row per (user, item) with a `<source>_rank` column per source
    (null = not retrieved by that source). The ranks are strong relevance
    features for the reranker; keeping them is what lets a larger candidate pool help.
    """
    s = context.schema
    names = [src.name for src in sources]
    parts = [src.generate(transactions, context).select(s.user, s.item, "rank", "source") for src in sources]
    unioned = reduce(lambda a, b: a.unionByName(b), parts)
    pivoted = unioned.groupBy(s.user, s.item).pivot("source", names).agg(F.min("rank"))
    for n in names:
        pivoted = pivoted.withColumnRenamed(n, f"{n}_rank")
    return pivoted
