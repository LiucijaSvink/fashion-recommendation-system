"""One-call tables for the notebook: each takes a scored fold, returns a frame."""
from __future__ import annotations

from typing import TYPE_CHECKING



import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..config import PipelineConfig

if TYPE_CHECKING:
    from ..pipeline.evaluation import ScoredFold


from .business import activity_segments, business_inputs, business_panel, segment_report

# ---------------------------------------------------------------------------
# Notebook-facing tables
#
# Thin wrappers that take a scored fold and return one table. They exist so a
# notebook cell is a single call with no unpacking, joining or date arithmetic
# in it — the reporting logic lives here with everything else.
# ---------------------------------------------------------------------------


def accuracy_table(scored: ScoredFold, cfg: PipelineConfig) -> pd.DataFrame:
    """Model against the bestseller baseline, over the customers who actually bought.

    ~95% of customers buy nothing in a given week and score zero whatever is
    recommended, so a figure averaged over everyone describes the population more than
    the model. The whole-population number — the one the competition scores — is
    attached as `.attrs["kaggle_map"]`, since it is one number rather than a table.
    """
    from ..pipeline.evaluation import accuracy_metrics

    everyone, buyers = scored.customers, scored.buyers
    baseline = {customer: list(scored.baseline) for customer in everyone}

    table = pd.DataFrame({
        "baseline": accuracy_metrics(baseline, scored.truth, cfg, buyers),
        "model": accuracy_metrics(scored.recommendations, scored.truth, cfg, buyers),
    })
    table["lift"] = table["model"] / table["baseline"].replace(0, float("nan"))

    over_everyone = accuracy_metrics(scored.recommendations, scored.truth, cfg, everyone)
    table.attrs["kaggle_map"] = over_everyone[f"MAP@{cfg.eval_k}"]
    table.attrs["customers"] = len(everyone)
    table.attrs["buyers"] = len(buyers)
    return table


def business_table(spark: SparkSession, cfg: PipelineConfig,
                   scored: ScoredFold) -> pd.DataFrame:
    """The commercial reading of the scored week, model against the baseline.

    Most of these figures mean nothing alone — "20% catalogue coverage" only lands next
    to the baseline's 0.01%. What each metric means is documented on `business_panel`
    and explained in the notebook rather than carried as a column of prose.
    """
    inputs = business_inputs(spark, cfg, scored.fold, scored.buyers)
    baseline = {customer: list(scored.baseline) for customer in scored.customers}
    return pd.DataFrame({
        "model": business_panel(scored.recommendations, scored.truth, inputs, cfg),
        "baseline": business_panel(baseline, scored.truth, inputs, cfg),
    })


def segment_table(spark: SparkSession, cfg: PipelineConfig,
                  scored: ScoredFold) -> pd.DataFrame:
    """Accuracy per customer segment, split into who they are and how well it works.

    Columns are grouped: **population** says how many customers the segment holds and
    how many bought that week, **accuracy** says how well the ranking served them.
    Without the population half, a strong-looking segment might be four hundred people.
    """
    segments = activity_segments(spark, cfg, scored.fold)
    table = segment_report(scored.recommendations, scored.truth, segments, cfg).copy()
    table["buyer rate %"] = 100 * table["buyers"] / table["customers"]

    population = ["customers", "buyers", "buyer rate %"]
    accuracy = [column for column in table.columns if column not in population]
    ordered = table[population + accuracy]
    ordered.columns = pd.MultiIndex.from_tuples(
        [("population", column) for column in population]
        + [("accuracy", column) for column in accuracy]
    )
    return ordered
