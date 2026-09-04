"""What each retrieval source contributes, and what capping it would cost.

These are the measurements behind the retrieval settings in `PipelineConfig`:
which sources earn their place, how deep each is taken, and whether the union
is worth trimming. Nothing here runs in production."""
from __future__ import annotations




import pandas as pd
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from ..config import PipelineConfig


def _truth_pairs(val: DataFrame, cfg: PipelineConfig) -> tuple[DataFrame, int]:
    truth = val.select(cfg.schema.user, cfg.schema.item).distinct().cache()
    return truth, truth.count()


def rank_columns(candidates: DataFrame) -> list[str]:
    """The `<source>_rank` columns, one per retrieval source that ran."""
    return [c for c in candidates.columns if c.endswith("_rank")]

def per_source_report(
    candidates: DataFrame, val: DataFrame, cfg: PipelineConfig,
    customers: DataFrame | None = None,
) -> pd.DataFrame:
    """Contribution of each source, plus the union's recall ceiling.

    `recall %` is the share of held-out (customer, item) purchases a source surfaces.
    Sources overlap, so their recalls do not sum to the union's.

    The held-out week is joined once and every source's hits are counted from that
    single pass; the per-source volume counts read two columns each, which parquet
    prunes cheaply.
    """
    user, item = cfg.schema.user, cfg.schema.item
    truth, n_truth = _truth_pairs(val, cfg)
    ranks = rank_columns(candidates)

    hit = F.col("_hit").isNotNull()
    hits = (
        candidates.join(truth.withColumn("_hit", F.lit(1)), on=[user, item], how="left")
        .agg(
            *[F.count(F.when(F.col(column).isNotNull() & hit, 1)).alias(column)
              for column in ranks],
            F.count(F.when(hit, 1)).alias("UNION"),
        )
        .collect()[0]
    )

    rows = []
    for column in ranks:
        volume = (candidates.filter(F.col(column).isNotNull())
                  .agg(F.count("*").alias("candidates"),
                       F.countDistinct(user).alias("customers")).collect()[0])
        rows.append({"source": column[:-5], "candidates": volume["candidates"],
                     "customers": volume["customers"], "hits": hits[column]})

    union = candidates.agg(F.count("*").alias("candidates"),
                           F.countDistinct(user).alias("customers")).collect()[0]
    rows.append({"source": "UNION", "candidates": union["candidates"],
                 "customers": union["customers"], "hits": hits["UNION"]})
    truth.unpersist()

    report = pd.DataFrame(rows).set_index("source")
    report["cands/customer"] = report["candidates"] / report["customers"]
    report["recall %"] = 100 * report["hits"] / n_truth
    report.attrs["val_pairs"] = n_truth
    if customers is not None:
        total = customers.select(user).distinct().count()
        report.attrs["customer_coverage %"] = 100 * union["customers"] / total
    return report[["candidates", "customers", "cands/customer", "hits", "recall %"]]


def rank_sweep(
    candidates: DataFrame, val: DataFrame, cfg: PipelineConfig,
    ks: tuple[int, ...] = (10, 25, 50, 100),
) -> pd.DataFrame:
    """Hits contributed by each source when truncated at rank <= K.

    The elbow — where a source stops adding hits — is what each source's `top_n` is set
    against. Ranks survive the union, so this needs no candidate regeneration.

    Every source x K combination is counted in one pass. The previous shape joined the
    held-out week once per combination: twenty scans to answer one question.
    """
    user, item = cfg.schema.user, cfg.schema.item
    truth, n_truth = _truth_pairs(val, cfg)
    ranks = rank_columns(candidates)

    hit = F.col("_hit").isNotNull()
    counts = (
        candidates.join(truth.withColumn("_hit", F.lit(1)), on=[user, item], how="left")
        .agg(*[F.count(F.when((F.col(column) <= k) & hit, 1)).alias(f"{column}@{k}")
               for column in ranks for k in ks])
        .collect()[0]
    )
    truth.unpersist()

    # reported as recall %, so this table is directly comparable with the other two —
    # counts alone say nothing about how much of the week a depth actually reaches
    sweep = pd.DataFrame(
        [{"source": column[:-5],
          **{f"K<={k}": 100 * counts[f"{column}@{k}"] / n_truth for k in ks}}
         for column in ranks]
    ).set_index("source")
    sweep.attrs["val_pairs"] = n_truth
    return sweep


def cap_sweep(
    candidates: DataFrame, val: DataFrame, cfg: PipelineConfig,
    caps: tuple[int | None, ...] = (60, 120, 180, None),
) -> pd.DataFrame:
    """Recall and candidate volume at several per-customer caps.

    Answers "does a bigger candidate budget buy anything?" — the trade being recall
    against the number of rows the reranker has to score. `None` measures the uncapped
    union, the reference every capped row is giving something up against.

    Every customer holding any candidate holds one at rank 1, so the customer count does
    not vary with the cap and is computed once instead of once per row of the table.
    """
    user, item = cfg.schema.user, cfg.schema.item
    truth, n_truth = _truth_pairs(val, cfg)

    ordering = F.col("n_sources").desc(), F.col(item)
    ranked = candidates.withColumn(
        "_cap_rank", F.row_number().over(Window.partitionBy(user).orderBy(*ordering))
    )

    def within(cap):
        return F.lit(True) if cap is None else F.col("_cap_rank") <= cap

    hit = F.col("_hit").isNotNull()
    totals = (
        ranked.join(truth.withColumn("_hit", F.lit(1)), on=[user, item], how="left")
        .agg(
            F.countDistinct(user).alias("customers"),
            *[F.count(F.when(within(cap), 1)).alias(f"rows@{cap}") for cap in caps],
            *[F.count(F.when(within(cap) & hit, 1)).alias(f"hits@{cap}") for cap in caps],
        )
        .collect()[0]
    )
    truth.unpersist()

    customers = totals["customers"]
    return pd.DataFrame([
        {"cap": cap if cap is not None else "uncapped",
         "rows": totals[f"rows@{cap}"],
         "cands/customer": totals[f"rows@{cap}"] / customers,
         "hits": totals[f"hits@{cap}"],
         "recall %": 100 * totals[f"hits@{cap}"] / n_truth}
        for cap in caps]).set_index("cap")
