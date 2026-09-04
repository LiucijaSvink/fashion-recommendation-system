"""Stage 2 — candidate retrieval.

Diagnostics for choosing sources and caps live in `recolib.analysis`, not here.

Five complementary sources are unioned into one row per (customer, item) carrying a
`<source>_rank` column each, then capped per customer. The cap is label-agnostic: it
orders by how many sources found a candidate, never by anything derived from the
held-out week.
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from ..retrieval import (ALSGraph, ItemCF, ProductCode, Repurchase,
                          RetrievalContext, SegmentPopularity, union)
from ..retrieval.base import CandidateSource
from ..config import PipelineConfig
from ..features import source_flags
from .io import FoldInputs, load_fold

log = logging.getLogger(__name__)


def build_sources(cfg: PipelineConfig,
                  neighbor_table: DataFrame | None = None) -> list[CandidateSource]:
    """The production source list. One definition, used by every stage and the CLI.

    `neighbor_table` lets a caller supply an already-computed itemcf neighbour matrix,
    so retrieval and the feature stage share one rather than each paying for it.
    """
    return [
        Repurchase(top_n=cfg.repurchase_top_n),
        ItemCF(top_n=cfg.itemcf_top_n, neighbors=cfg.itemcf_neighbors,
               seed_weeks=cfg.seed_weeks, neighbor_table=neighbor_table),
        ProductCode(top_n=cfg.product_code_top_n, seed_weeks=cfg.seed_weeks),
        SegmentPopularity(top_n=cfg.popularity_top_n, weeks=cfg.popularity_weeks,
                          global_fallback="no_info_only", name="popularity"),
        ALSGraph(top_n=cfg.als_top_n, rank=cfg.als_rank, iters=cfg.als_iters,
                 seed=cfg.seed),
    ]


def source_names(cfg: PipelineConfig) -> list[str]:
    return [source.name for source in build_sources(cfg)]


def _cap_per_customer(candidates: DataFrame, cfg: PipelineConfig, names: list[str]) -> DataFrame:
    """Keep the top `max_candidates` per customer, most-corroborated first.

    `max_candidates=None` keeps the whole union — the shipped setting, because section
    3.3 of the notebook measures capping as a loss of recall that buys nothing back.
    """
    user = cfg.schema.user
    if cfg.max_candidates is None:
        return source_flags(candidates, names)
    order = Window.partitionBy(user).orderBy(
        F.col("n_sources").desc(), F.col(cfg.schema.item)
    )
    return (
        source_flags(candidates, names)
        .withColumn("_rank", F.row_number().over(order))
        .filter(F.col("_rank") <= cfg.max_candidates)
        .drop("_rank")
    )


def build_candidates(
    spark: SparkSession,
    cfg: PipelineConfig,
    fold: str,
    *,
    inputs: FoldInputs | None = None,
) -> DataFrame:
    """Generate, union, cap and persist this fold's candidate set."""
    inputs = inputs or load_fold(spark, cfg, fold)
    context = RetrievalContext(
        spark=spark, customers=inputs.customers, articles=inputs.articles,
        train_end=inputs.train_end, full_history=inputs.history, schema=cfg.schema,
    )
    # the itemcf neighbour matrix is needed twice — here and by ItemCFScore in the
    # feature stage — so it is computed once and persisted beside the fold
    itemcf = ItemCF(top_n=cfg.itemcf_top_n, neighbors=cfg.itemcf_neighbors,
                    seed_weeks=cfg.seed_weeks)
    neighbors_path = cfg.path(fold, "_itemcf_neighbors")
    itemcf.neighbor_table(inputs.train, context).write.mode("overwrite").parquet(neighbors_path)
    log.info("%s: wrote itemcf neighbours -> %s", fold, neighbors_path)

    sources = build_sources(cfg, neighbor_table=spark.read.parquet(neighbors_path))
    names = [source.name for source in sources]

    candidates = _cap_per_customer(union(sources, inputs.train, context), cfg, names)
    out_path = cfg.path(fold, "candidates")
    candidates.write.mode("overwrite").parquet(out_path)
    log.info("%s: wrote candidates -> %s", fold, out_path)
    return spark.read.parquet(out_path)
