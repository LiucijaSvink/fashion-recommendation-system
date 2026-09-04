"""Diagnostics and reporting that inform the pipeline without being part of it.
Split by question rather than by mechanism:
* `retrieval` — what each candidate source contributes, and what capping costs
* `folds`     — repeating a measurement across folds; leakage-free evaluation
* `business`  — what a set of recommendations would mean commercially
* `tables`    — one-call tables for the notebook
"""
from .business import (BusinessInputs, ACTIVE, COLD, DORMANT, LAPSED, activity_segments,
                       business_inputs, business_panel, segment_report)
from .plots import feature_importance_chart, segment_chart
from .folds import across_folds, development_metrics, fold_inputs, mean_sd
from .retrieval import cap_sweep, per_source_report, rank_columns, rank_sweep
from .tables import accuracy_table, business_table, segment_table
__all__ = [
    "per_source_report", "rank_sweep", "cap_sweep", "rank_columns",
    "across_folds", "development_metrics", "fold_inputs", "mean_sd", 
    "business_inputs", "business_panel", "segment_report", "activity_segments",
    "BusinessInputs", "ACTIVE", "LAPSED", "DORMANT", "COLD",
    "accuracy_table", "business_table", "segment_table",
    "feature_importance_chart", "segment_chart",
]
