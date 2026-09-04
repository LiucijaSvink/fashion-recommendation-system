"""Stage 3 — feature engineering.

Each `FeatureSource` returns a small block plus its join keys; the candidate table is
left-joined against each in turn. Labels come from the fold's held-out week (all zeros
for the submission fold, which has no future to look at).
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PipelineConfig
from ..features import (ArticleMetadata, CategoryAffinity, CustomerMetadata,
                        FeatureContext, ItemAggregates, ItemBuyerDemographic,
                        ItemCFScore, UserAggregates, UserItemHistory, attach_labels,
                        derived_ratios)
from ..features.base import FeatureSource
from .io import FoldInputs, load_fold

log = logging.getLogger(__name__)

# `CategoryAffinity` returns a block keyed on a raw article attribute, so the candidate
# table has to carry those attributes while the affinity sources run. Two of them
# (product_type_no, department_no) are also features, but as the cast codes that
# `ArticleMetadata` produces — so the raw versions are dropped just before it runs, and
# the two that are only ever join keys are dropped at the end.
_JOIN_ONLY_COLUMNS = ("product_code", "product_group_name")


def build_feature_sources(cfg: PipelineConfig) -> list[FeatureSource]:
    """The production feature list. Adding an affinity is one entry in `cfg.affinity_keys`."""
    return [
        UserAggregates(recent_weeks=cfg.user_recent_weeks),
        ItemAggregates(windows=cfg.item_windows),
        *[CategoryAffinity(key, recent_weeks=cfg.user_recent_weeks, out_name=name)
          for key, name in cfg.affinity_keys],
        UserItemHistory(),
        ItemCFScore(seed_weeks=cfg.seed_weeks),
        ItemBuyerDemographic(),
        CustomerMetadata(),
        ArticleMetadata(),
    ]


def _load_neighbors(spark: SparkSession, cfg: PipelineConfig, fold: str) -> DataFrame:
    """The itemcf neighbour matrix persisted by the retrieval stage, for `ItemCFScore`."""
    path = cfg.path(fold, "_itemcf_neighbors")
    try:
        return spark.read.parquet(path)
    except Exception as error:                       # noqa: BLE001 - re-raised with context
        raise FileNotFoundError(
            f"{path} is missing — run the 'candidates' stage for {fold} first; it writes "
            "the neighbour matrix that ItemCFScore reads."
        ) from error


def build_features(
    spark: SparkSession,
    cfg: PipelineConfig,
    fold: str,
    *,
    inputs: FoldInputs | None = None,
    candidates: DataFrame | None = None,
) -> DataFrame:
    """Join every feature block onto the fold's candidates, attach labels, persist.

    The join runs in `cfg.feature_chunks` passes over slices of the candidate table,
    hashed by customer. Feature blocks are materialised once beforehand and reused by
    every pass, so the numbers are identical to a single-pass build — what changes is
    that each pass shuffles a fraction of the rows.

    That matters because a single pass over ~200M candidates is eight joins' worth of
    shuffle held at once, which exhausts local disk on a workstation. Hashing on the
    customer keeps every customer's rows inside one pass, so per-customer features and
    ranks are unaffected by where the boundaries fall.
    """
    inputs = inputs or load_fold(spark, cfg, fold)
    candidates = candidates if candidates is not None else spark.read.parquet(
        cfg.path(fold, "candidates")
    )
    context = FeatureContext(
        spark=spark, end=inputs.train_end, transactions=inputs.train,
        history=inputs.history, customers=inputs.customers, articles=inputs.articles,
        neighbors=_load_neighbors(spark, cfg, fold), schema=cfg.schema,
    )

    blocks = _materialise_blocks(spark, cfg, fold, context)

    user = cfg.schema.user
    affinity_keys = [key for key, _ in cfg.affinity_keys]
    article_keys = inputs.articles.select(cfg.schema.item, *affinity_keys)
    out_path = cfg.path(fold, "features")
    chunks = max(1, cfg.feature_chunks)

    for chunk in range(chunks):
        slice_ = (candidates if chunks == 1 else
                  candidates.filter(F.pmod(F.hash(F.col(user)), F.lit(chunks)) == chunk))
        features = slice_.join(article_keys, on=cfg.schema.item, how="left")
        for source, block in blocks:
            if isinstance(source, ArticleMetadata):
                features = features.drop(*affinity_keys)
            features = features.join(block, on=source.join_keys, how="left")

        features = derived_ratios(attach_labels(features, inputs.val, cfg.schema))
        features = features.drop(*_JOIN_ONLY_COLUMNS)
        mode = "overwrite" if chunk == 0 else "append"
        features.write.mode(mode).parquet(out_path)
        log.info("%s: wrote features chunk %s/%s", fold, chunk + 1, chunks)

    _clear_block_cache(spark, cfg, fold)
    log.info("%s: wrote features -> %s", fold, out_path)
    return spark.read.parquet(out_path)


def _block_path(cfg: PipelineConfig, fold: str, source) -> str:
    return f"{cfg.fold_dir(fold)}/_blocks/{source.name}"


def _materialise_blocks(spark: SparkSession, cfg: PipelineConfig, fold: str, context):
    """Compute each feature block once and persist it.

    Without this, every chunk recomputes every aggregate from the full history — the
    blocks are small, the inputs are not.
    """
    blocks = []
    for source in build_feature_sources(cfg):
        block, keys = source.compute(context)
        path = _block_path(cfg, fold, source)
        block.write.mode("overwrite").parquet(path)
        source.join_keys = keys
        blocks.append((source, spark.read.parquet(path)))
        log.info("%s: materialised block %s", fold, source.name)
    return blocks


def _clear_block_cache(spark: SparkSession, cfg: PipelineConfig, fold: str) -> None:
    """Remove the materialised blocks — they are scratch, not an artifact."""
    import shutil
    shutil.rmtree(f"{cfg.fold_dir(fold)}/_blocks", ignore_errors=True)


def feature_summary(features: DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Grouped column inventory — what the reranker actually sees."""
    import pandas as pd

    keys = {cfg.schema.user, cfg.schema.item, "label"}
    groups: dict[str, list[str]] = {}
    for column in features.columns:
        if column in keys:
            continue
        if column.endswith("_rank") or column.startswith("src_") or column == "n_sources":
            group = "retrieval"
        elif column.startswith("user_item") or column in {"price_ratio", "age_gap",
                                                          "bought_before", "user_dept_ratio"}:
            group = "cross"
        elif column.startswith("user_") or column in {"age", "is_active"} or column.endswith("_idx"):
            group = "customer"
        else:
            group = "item"
        groups.setdefault(group, []).append(column)

    return pd.DataFrame(
        [{"group": g, "n": len(cols), "columns": ", ".join(sorted(cols))}
         for g, cols in sorted(groups.items())]
    ).set_index("group")
