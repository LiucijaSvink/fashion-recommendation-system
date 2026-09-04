"""End-to-end orchestration: raw CSV → folds → candidates → features → model → submission.

Each stage is an ordinary function taking `(spark, cfg, ...)`, so the notebook, the CLI
and the tests all drive exactly the same code.

    from recolib.config import PipelineConfig
    from recolib.pipeline import run_all
    run_all(spark, PipelineConfig.from_json("configs/config.json"))
"""
from __future__ import annotations

import logging

from pyspark.sql import SparkSession

from ..config import PipelineConfig
from ..modeling import LambdaRanker
from .data import build_folds, fold_table, prepare
from .evaluation import evaluate, ground_truth, metrics_table, sample_customers, score
from .featurize import build_features, build_feature_sources, feature_summary
from .io import FoldInputs, load_fold
from .candidates import build_candidates, build_sources, source_names
from .submit import write_submission
from .training import build_training_frame, train

log = logging.getLogger(__name__)

STAGES = ("prepare", "candidates", "features", "train", "evaluate", "submit")

__all__ = [
    "STAGES", "run_all", "run_fold",
    "prepare", "build_folds", "fold_table",
    "load_fold", "FoldInputs",
    "build_candidates", "build_sources", "source_names",
    "build_features", "build_feature_sources", "feature_summary",
    "train", "build_training_frame",
    "evaluate", "score", "ground_truth", "sample_customers", "metrics_table",
    "write_submission",
]


def run_fold(spark: SparkSession, cfg: PipelineConfig, fold: str) -> None:
    """Retrieval + features for one fold (the two expensive, per-fold stages)."""
    inputs = load_fold(spark, cfg, fold)
    candidates = build_candidates(spark, cfg, fold, inputs=inputs)
    build_features(spark, cfg, fold, inputs=inputs, candidates=candidates)


def run_all(
    spark: SparkSession,
    cfg: PipelineConfig,
    *,
    stages: tuple[str, ...] = STAGES,
    overwrite: bool = False,
) -> LambdaRanker | None:
    """Run the pipeline. `stages` selects which parts to execute."""
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; valid: {list(STAGES)}")

    folds = list(cfg.train_folds) + [cfg.submission_fold]
    if "prepare" in stages:
        prepare(spark, cfg, overwrite=overwrite)
    if "candidates" in stages:
        for fold in folds:
            build_candidates(spark, cfg, fold)
    if "features" in stages:
        for fold in folds:
            build_features(spark, cfg, fold)

    model = None
    if "train" in stages:
        model, _ = train(spark, cfg)
    if "evaluate" in stages:
        if model is None:
            raise ValueError("evaluate needs a model — include the 'train' stage")
        holdout = cfg.holdout_fold
        if holdout is None:
            log.warning(
                "skipping evaluate: train_folds %s includes the newest fold, so every "
                "fold available to score is one the model trained on. Train on older "
                "folds to get an out-of-sample number.", list(cfg.train_folds),
            )
        else:
            log.info("%s (held out) -> %s", holdout, evaluate(spark, cfg, model, holdout))
    if "submit" in stages:
        if model is None:
            raise ValueError("submit needs a model — include the 'train' stage")
        write_submission(spark, cfg, model)
    return model
