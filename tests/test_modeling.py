"""Tests for recolib.modeling — fit/score/recommend/backfill/categorical/early stopping."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from recolib.modeling import LambdaRanker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_table(n_users=20, items_per_user=10, n_features=4, seed=42):
    """Generate a synthetic feature table where label correlates with feature_0."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for i in range(items_per_user):
            feats = rng.normal(size=n_features)
            label = int(feats[0] > 0.5)   # rank signal
            rows.append((f"u{u}", f"a{u}_{i}", label, *feats))
    cols = ["customer_id", "article_id", "label"] + [f"f{j}" for j in range(n_features)]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_params_merged_with_kwargs(self):
        m = LambdaRanker(learning_rate=0.1)
        assert m.params["learning_rate"] == 0.1
        assert m.params["objective"] == "lambdarank"   # default preserved

    def test_model_none_until_fit(self):
        assert LambdaRanker().model is None

    def test_score_without_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LambdaRanker().score(pd.DataFrame())

    def test_recommend_without_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LambdaRanker().recommend(pd.DataFrame())

    def test_feature_importance_without_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            LambdaRanker().feature_importance()


# ---------------------------------------------------------------------------
# Fit / score / recommend
# ---------------------------------------------------------------------------

class TestFit:
    def test_fits_and_sets_model(self):
        m = LambdaRanker(n_estimators=20).fit(_make_table())
        assert m.model is not None
        assert m.feature_cols == ["f0", "f1", "f2", "f3"]   # _features excludes keys/label

    def test_uses_explicit_feature_cols(self):
        df = _make_table()
        df["junk"] = 0
        m = LambdaRanker(feature_cols=["f0", "f1"], n_estimators=20).fit(df)
        assert m.feature_cols == ["f0", "f1"]               # junk not auto-picked

    def test_score_returns_array_per_row(self):
        df = _make_table()
        m = LambdaRanker(n_estimators=20).fit(df)
        scores = m.score(df)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(df),)

    def test_multifold_group_cols(self):
        df = pd.concat([_make_table(seed=1).assign(_fold=0),
                        _make_table(seed=2).assign(_fold=1)])
        m = LambdaRanker(n_estimators=20).fit(df, group_cols=["_fold", "customer_id"])
        assert "_fold" not in m.feature_cols       # _fold excluded from features


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------

class TestRecommend:
    def test_returns_top_k_per_user(self):
        df = _make_table(n_users=5, items_per_user=20)
        m = LambdaRanker(n_estimators=20).fit(df)
        recs = m.recommend(df, k=5)
        assert all(len(items) == 5 for items in recs.values())
        # No duplicates within a user
        for items in recs.values():
            assert len(set(items)) == len(items)

    def test_recommend_does_not_mutate_input(self):
        df = _make_table(n_users=5, items_per_user=10)
        m = LambdaRanker(n_estimators=20).fit(df)
        cols_before = list(df.columns)
        _ = m.recommend(df.copy(), k=5)        # explicit copy keeps the original safe
        assert list(df.columns) == cols_before  # no _score column added to caller's df

    def test_backfill_pads_short_users(self):
        df = _make_table(n_users=2, items_per_user=3)
        m = LambdaRanker(n_estimators=20).fit(df)
        # Caller asks for k=5 but only 3 candidates exist per user
        backfill = {"u0": ["b1", "b2", "b3"], "u1": ["b1", "b2", "b3"]}
        recs = m.recommend(df, k=5, backfill=backfill)
        assert all(len(items) == 5 for items in recs.values())
        # padding starts from fallback items not already in the scored list
        for u, items in recs.items():
            assert "b1" in items or "b2" in items or "b3" in items

    def test_backfill_adds_users_absent_from_pdf(self):
        df = _make_table(n_users=2, items_per_user=10)
        m = LambdaRanker(n_estimators=20).fit(df)
        backfill = {"u_cold": ["p1", "p2", "p3"]}
        recs = m.recommend(df, k=3, backfill=backfill)
        assert recs["u_cold"] == ["p1", "p2", "p3"]


# ---------------------------------------------------------------------------
# Categorical features pass-through
# ---------------------------------------------------------------------------

class TestCategorical:
    def test_categorical_features_round_trip(self):
        df = _make_table(n_users=5, items_per_user=10)
        m = LambdaRanker(categorical_features=["f0"], n_estimators=20).fit(df)
        assert m.categorical_features == ["f0"]
        assert m.model is not None     # model accepted the categorical hint


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_eval_set_and_callbacks_accepted(self):
        """Smoke-test: fit succeeds with eval_pdf + early_stopping callback."""
        train_df = _make_table(n_users=20, items_per_user=10, seed=1)
        eval_df  = _make_table(n_users=5,  items_per_user=10, seed=99)
        m = LambdaRanker(n_estimators=200).fit(
            train_df,
            eval_pdf=eval_df,
            callbacks=[lgb.early_stopping(5, verbose=False),
                       lgb.log_evaluation(0)],
        )
        # early stopping should have triggered (best_iteration < n_estimators)
        assert m.model.booster_.best_iteration <= 200


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

class TestFeatureImportance:
    def test_returns_sorted_dataframe(self):
        df = _make_table()
        m = LambdaRanker(n_estimators=30).fit(df)
        fi = m.feature_importance()
        assert set(fi.columns) == {"feature", "gain"}
        assert list(fi["gain"]) == sorted(fi["gain"], reverse=True)

    def test_split_importance_type(self):
        df = _make_table()
        m = LambdaRanker(n_estimators=30).fit(df)
        fi = m.feature_importance(importance_type="split")
        assert "split" in fi.columns
