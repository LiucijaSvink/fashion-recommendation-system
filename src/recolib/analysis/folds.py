"""Repeating a measurement across folds, and evaluating without leakage."""
from __future__ import annotations

import logging


import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from ..cache import cached
from ..config import PipelineConfig

log = logging.getLogger(__name__)


from .retrieval import per_source_report  # noqa: F401  (re-exported for convenience)

# ---------------------------------------------------------------------------
# Repeating a measurement across folds
#
# One validation week is one week — the EDA measured weekly volume swinging by a factor
# of three, so a number from a single week is not a result. Every experiment here is run
# on each fold and reported as a mean with the spread across folds; a setting is only
# chosen when the folds agree.
# ---------------------------------------------------------------------------


def fold_inputs(
    spark: SparkSession, cfg: PipelineConfig, fold: str, *, sample_users: int | None = None
) -> tuple[DataFrame, DataFrame]:
    """`(features, val)` for one fold, optionally cut to a deterministic customer sample.

    Sampling is there to make a six-fold sweep affordable: recalls are ratios, so a
    fixed sample moves the precision of the estimate, not what it estimates.
    """

    features = spark.read.parquet(cfg.path(fold, "features"))
    val = spark.read.parquet(cfg.path(fold, "val"))
    if sample_users:
        from ..pipeline.evaluation import sample_customers

        users = sample_customers(features, cfg, sample_users)
        if sample_users <= 50_000:          # small enough to sit on every executor
            users = F.broadcast(users)
        user = cfg.schema.user
        features, val = features.join(users, on=user), val.join(users, on=user)
    return features, val


def across_folds(
    measure,
    spark: SparkSession,
    cfg: PipelineConfig,
    folds: list[str] | None = None,
    *,
    sample_users: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Run a single-fold measurement on every fold; return the mean, spread attached.

    `measure` is any of `per_source_report`, `rank_sweep`, `cap_sweep` — anything with
    the signature `(features, val, cfg, **kwargs)` returning an indexed frame. The
    per-fold detail survives on `.attrs["per_fold"]`, the spread on `.attrs["sd"]`.
    """
    folds = list(folds or cfg.fold_names)
    # DataFrame arguments cannot go in a cache key (their repr changes every run), so
    # the signature is built from the scalar arguments that actually change results
    signature = {name: value for name, value in kwargs.items()
                 if isinstance(value, (int, float, str, bool, tuple, list, type(None)))}
    signature["sample_users"] = sample_users

    frames, val_pairs = {}, {}
    for fold in folds:
        def run(fold=fold):
            features, val = fold_inputs(spark, cfg, fold, sample_users=sample_users)
            result = measure(features, val, cfg, **kwargs)
            result.attrs.setdefault("val_pairs", None)
            return result

        result = cached(cfg, f"{measure.__name__}.{fold}", run, extra=signature)
        frames[fold] = result
        val_pairs[fold] = result.attrs.get("val_pairs")

    per_fold = pd.concat(frames, names=["fold"])
    grouped = per_fold.groupby(level=-1, sort=False)
    order = frames[folds[0]].index
    mean, sd = grouped.mean().reindex(order), grouped.std().reindex(order)
    mean.attrs.update(folds=folds, sd=sd, per_fold=per_fold, val_pairs=val_pairs,
                      sample_users=sample_users)
    return mean


def mean_sd(result: pd.DataFrame, columns: list[str] | None = None,
            decimals: int = 2) -> pd.DataFrame:
    """Format an `across_folds` result as `mean ± sd` — the fold spread made visible."""
    sd = result.attrs.get("sd")
    if sd is None:
        return result.round(decimals)
    cols = columns or list(result.columns)
    formatted = pd.DataFrame(index=result.index)
    for col in cols:
        formatted[col] = [
            # one fold has no spread to report; NaN would read as a broken number
            f"{mean:,.{decimals}f}" if pd.isna(spread)
            else f"{mean:,.{decimals}f} ± {spread:,.{decimals}f}"
            for mean, spread in zip(result[col], sd[col])
        ]
    return formatted


def development_metrics(spark: SparkSession, cfg: PipelineConfig) -> pd.DataFrame:
    """Score each development week and average, model against the bestseller baseline.

    A week supplies its own labels, so a model trained on it cannot be judged on it. Each
    development week is scored by a model fitted on the weeks behind it — an expanding
    window, never the future. The oldest week has nothing behind it and is skipped.

    Figures are restricted to customers who bought that week, exactly as the holdout
    table is, so the two can be read side by side. A non-buyer contributes zero to every
    metric here, so this is the same measurement rescaled rather than a different one.

    The models are deliberately unequal: the most recent development week has four weeks
    of training behind it, the oldest has one. A lower score on an older week can mean a
    harder week or less training data. Per-fold detail is on `.attrs["per_fold"]`.
    """
    from ..pipeline.evaluation import (accuracy_metrics, baseline_items, evaluate,
                                       ground_truth)
    from ..pipeline.training import train

    names = cfg.fold_names
    models, baselines, context = {}, {}, {}
    for fold in cfg.train_folds:
        older = tuple(names[names.index(fold) + 1:])
        if not older:
            log.info("%s: no older week to train on — skipped", fold)
            continue
        run_cfg = cfg.with_(train_folds=older)

        def score(run_cfg=run_cfg, fold=fold, older=older):
            model, frame = train(spark, run_cfg)
            return {"training weeks": len(older), "training rows": len(frame),
                    **evaluate(spark, run_cfg, model, fold)}

        def baseline(fold=fold):
            val = spark.read.parquet(cfg.path(fold, "val"))
            truth = ground_truth(val, cfg)
            everyone = [row[cfg.schema.user] for row in
                        spark.read.parquet(cfg.customers_path)
                        .select(cfg.schema.user).distinct().collect()]
            recommended = baseline_items(spark, cfg, fold)
            return accuracy_metrics({customer: list(recommended) for customer in everyone},
                                    truth, cfg, everyone)

        models[fold] = cached(run_cfg, f"development_metrics.{fold}", score)
        baselines[fold] = cached(cfg, f"development_baseline.{fold}", baseline)
        context[fold] = {"training weeks": len(older),
                         "buyers": models[fold]["buyers"]}
        log.info("%s scored, trained on %s", fold, list(older))

    model_frame = pd.DataFrame(models)
    baseline_frame = pd.DataFrame(baselines)
    metrics = [m for m in baseline_frame.index if m in model_frame.index]

    # Restrict to buyers, matching the holdout table. Every metric here is a mean over
    # customers in which a non-buyer contributes exactly zero — there is nothing they
    # could have been shown that would count — so averaging over buyers alone is the
    # same measurement rescaled by customers/buyers, not a different one.
    scale = pd.Series({fold: values["customers scored"] / values["buyers"]
                       for fold, values in models.items()})
    model_frame = model_frame.loc[metrics].mul(scale, axis=1)
    baseline_frame = baseline_frame.loc[metrics].mul(scale, axis=1)

    # lift is computed per week and then averaged, so it carries a spread of its own —
    # a ratio of two averages would hide how much it moves from week to week
    lift_frame = model_frame / baseline_frame.replace(0, float("nan"))

    def averaged(frame: pd.DataFrame) -> pd.Series:
        return pd.Series({metric: f"{frame.loc[metric].mean():.4f} "
                                  f"({frame.loc[metric].std():.4f})" for metric in metrics})

    table = pd.DataFrame({"baseline": averaged(baseline_frame),
                          "model": averaged(model_frame),
                          "lift": averaged(lift_frame)})
    table.attrs["context"] = pd.DataFrame.from_dict(context, orient="index")
    table.attrs["per_fold"] = model_frame
    return table
