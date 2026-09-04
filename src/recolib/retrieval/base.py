"""Candidate-retrieval base class + shared context.

Every retrieval strategy is a `CandidateSource` that turns a transactions
DataFrame into ranked candidates: columns (user, item, rank, source), where
`rank` (1 = best) reflects that source's own relevance ordering. Column names
come from the `Schema` on `RetrievalContext`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from pyspark.sql import DataFrame, SparkSession

from ..schema import DEFAULT_SCHEMA, Schema


@dataclass
class RetrievalContext:
    """Shared inputs passed to every source's `.generate`."""
    spark: SparkSession
    customers: DataFrame                 # at least [schema.user]
    articles: DataFrame                  # at least [schema.item, product_code, ...]
    train_end: date                      # cutoff; no feature/candidate sees data after this
    full_history: DataFrame | None = None  # all transactions <= train_end (for lifetime signals)
    schema: Schema = DEFAULT_SCHEMA


class CandidateSource(ABC):
    """Abstract retrieval strategy. Subclasses set `name` and implement `generate`."""

    name: str = "base"

    def __init__(self, top_n: int = 100):
        self.top_n = top_n

    @abstractmethod
    def generate(self, transactions: DataFrame, context: RetrievalContext) -> DataFrame:
        """Return candidates as (user, item, rank, source)."""
        raise NotImplementedError

    def _tag(self, df: DataFrame) -> DataFrame:
        """Attach the `source` column (call at the end of `generate`)."""
        from pyspark.sql import functions as F
        tagged = df.withColumn("source", F.lit(self.name))
        return tagged

    def __repr__(self) -> str:
        return f"{type(self).__name__}(top_n={self.top_n})"
