"""FeatureSource ABC + shared FeatureContext.

A `FeatureSource` produces a small dataframe of related feature columns plus
the join keys to attach it to the candidate set. The reranker pipeline composes
sources by joining each one in turn — see `sources.py` for concrete subclasses.

Mirrors `CandidateSource`: same strategy-pattern shape, different stage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from pyspark.sql import DataFrame, SparkSession

from ..schema import DEFAULT_SCHEMA, Schema


@dataclass
class FeatureContext:
    """Inputs shared across feature sources.

    Each source uses 1–3 of these fields; the bundle exists so every
    `compute(context)` has the same signature (the trade-off for uniform
    polymorphism — same as `RetrievalContext`).

    Fields
    ------
    spark        : active SparkSession
    end          : feature cutoff date; all aggregations must be ≤ this date
                   to avoid label leakage
    transactions : fold-recent training window (item popularity, itemcf seeds)
    history      : full lifetime transactions (active/inactive coalesce, repurchase)
    customers    : customer table (age, demographics)
    articles     : article catalog (categories, codes)
    neighbors    : itemcf neighbor matrix from the retrieval stage; required
                   only by `ItemCFScore`
    schema       : column-name mapping (default: Schema())
    """
    spark: SparkSession
    end: date
    transactions: DataFrame
    history: DataFrame
    customers: DataFrame
    articles: DataFrame
    neighbors: DataFrame | None = None
    schema: Schema = DEFAULT_SCHEMA


class FeatureSource(ABC):
    """A source of feature columns to join onto candidates.

    Subclasses declare:
      * `name`     — short identifier for diagnostics
      * `produces` — list of column names added (excluding join keys)
      * `compute`  — returns `(block_df, join_keys)` where `block_df` has the
                     join keys plus every column in `produces`

    For parameterized sources (e.g. `CategoryAffinity`) set `name` and
    `produces` in `__init__`; for fixed sources, set them as class attributes.
    """
    name: str
    produces: list[str]

    @abstractmethod
    def compute(self, context: FeatureContext) -> tuple[DataFrame, list[str]]:
        """Return `(block_df, join_keys)` — the columns to add and how to join them."""
