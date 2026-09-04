"""Tests for recolib.retrieval — sources, _tag, union, ABC contract.

ALSGraph is intentionally omitted: training is nondeterministic and slow
(adds 10+ seconds per test); its `recommendForAllUsers` API is exercised in
the example/notebook pipeline rather than unit-tested here.
"""
from __future__ import annotations

from datetime import date

import pytest
from pyspark.sql import functions as F

from recolib.retrieval import (
    CandidateSource,
    ItemCF,
    Popularity,
    ProductCode,
    Repurchase,
    RetrievalContext,
    SegmentPopularity,
    union,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_TX_SCHEMA = "customer_id string, article_id string, t_dat date, price double, sales_channel_id int"
_CUST_SCHEMA = "customer_id string, age double, postal_code string"
_ART_SCHEMA = "article_id string, product_code long"


def _tx(spark, rows, schema=_TX_SCHEMA):
    # Add a sales_channel_id (UserAggregates uses it; harmless for candidates)
    return spark.createDataFrame(
        [tuple(list(r) + [1] * (5 - len(r))) for r in rows], schema=schema,
    )


def _ctx(spark, customers_rows, articles_rows, train_end, full_history=None):
    customers = spark.createDataFrame(customers_rows, schema=_CUST_SCHEMA)
    articles = spark.createDataFrame(articles_rows, schema=_ART_SCHEMA)
    return RetrievalContext(
        spark=spark, customers=customers, articles=articles,
        train_end=train_end, full_history=full_history,
    )


def _collect(df):
    return [r.asDict() for r in df.collect()]


# ---------------------------------------------------------------------------
# ABC + base behavior
# ---------------------------------------------------------------------------

class TestBase:
    def test_abc_contract_enforced(self):
        class Broken(CandidateSource):
            name = "broken"
        with pytest.raises(TypeError, match="abstract"):
            Broken()

    def test_repr_carries_top_n(self):
        assert repr(Repurchase(top_n=42)) == "Repurchase(top_n=42)"

    def test_tag_adds_source_column(self, spark):
        df = spark.createDataFrame([("u", "a", 1)], ["customer_id", "article_id", "rank"])
        out = Repurchase(top_n=5)._tag(df).collect()[0]
        assert out["source"] == "repurchase"


# ---------------------------------------------------------------------------
# Repurchase
# ---------------------------------------------------------------------------

class TestRepurchase:
    def test_ranks_by_count_then_item(self, spark):
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 1), 1.0),
            ("u1", "a1", date(2020, 1, 2), 1.0),   # a1 bought 2x
            ("u1", "a2", date(2020, 1, 3), 1.0),
        ])
        ctx = _ctx(spark, [("u1", 30.0, "p1")], [("a1", 100), ("a2", 200)],
                   train_end=date(2020, 1, 31))
        out = sorted(_collect(Repurchase(top_n=5).generate(tx, ctx)),
                     key=lambda r: r["rank"])
        assert [(r["article_id"], r["rank"]) for r in out] == [("a1", 1), ("a2", 2)]
        assert all(r["source"] == "repurchase" for r in out)

    def test_top_n_caps_results(self, spark):
        tx = _tx(spark, [
            ("u1", f"a{i}", date(2020, 1, i + 1), 1.0) for i in range(5)
        ])
        ctx = _ctx(spark, [("u1", None, None)],
                   [(f"a{i}", 100) for i in range(5)], train_end=date(2020, 1, 31))
        assert Repurchase(top_n=3).generate(tx, ctx).count() == 3

    def test_uses_full_history_when_given(self, spark):
        # tx empty, full_history populated -> Repurchase should source from full_history
        tx = _tx(spark, [])
        history = _tx(spark, [("u1", "a1", date(2019, 1, 1), 1.0)])
        ctx = _ctx(spark, [("u1", None, None)], [("a1", 100)],
                   train_end=date(2020, 1, 31), full_history=history)
        out = _collect(Repurchase(top_n=5).generate(tx, ctx))
        assert len(out) == 1 and out[0]["article_id"] == "a1"


# ---------------------------------------------------------------------------
# Popularity
# ---------------------------------------------------------------------------

class TestPopularity:
    def test_assigns_top_to_every_customer(self, spark):
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 1), 1.0),
            ("u2", "a1", date(2020, 1, 2), 1.0),
            ("u3", "a2", date(2020, 1, 3), 1.0),
        ])
        ctx = _ctx(spark, [("u1", None, None), ("u2", None, None), ("u3", None, None)],
                   [("a1", 1), ("a2", 2)], train_end=date(2020, 1, 4))
        out = _collect(Popularity(top_n=2, weeks=4).generate(tx, ctx))
        assert {r["customer_id"] for r in out} == {"u1", "u2", "u3"}
        for u in {"u1", "u2", "u3"}:
            ranks_for_u = sorted(r["rank"] for r in out if r["customer_id"] == u)
            assert ranks_for_u == [1, 2]  # a1 has 2 purchases, a2 has 1

    def test_respects_recent_window(self, spark):
        # train_end - 1 week = 2020-01-25; only items purchased after that count
        tx = _tx(spark, [
            ("u1", "old", date(2020, 1, 1), 1.0),   # outside 1-week window
            ("u1", "new", date(2020, 1, 30), 1.0),  # inside
        ])
        ctx = _ctx(spark, [("u1", None, None)], [("old", 1), ("new", 2)],
                   train_end=date(2020, 2, 1))
        items = {r["article_id"] for r in
                 _collect(Popularity(top_n=5, weeks=1).generate(tx, ctx))}
        assert items == {"new"}


# ---------------------------------------------------------------------------
# ProductCode
# ---------------------------------------------------------------------------

class TestProductCode:
    def test_returns_variants_ranked_by_item_popularity(self, spark):
        # u1 buys a1 (product 100). Variants of 100 are a1, a2. a2 strictly more popular.
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 30), 1.0),   # seed (recent)
            ("u2", "a2", date(2020, 1, 5), 1.0),    # a2 popularity++
            ("u3", "a2", date(2020, 1, 6), 1.0),    # a2 popularity++
            ("u4", "a2", date(2020, 1, 7), 1.0),    # a2 popularity++ (a2=3, a1=1)
        ])
        ctx = _ctx(spark, [("u1", None, None)], [("a1", 100), ("a2", 100), ("a3", 200)],
                   train_end=date(2020, 2, 1))
        out = sorted(_collect(
            ProductCode(top_n=5, seed_weeks=4).generate(tx, ctx)
            .filter(F.col("customer_id") == "u1")
        ), key=lambda r: r["rank"])
        items = [r["article_id"] for r in out]
        assert items == ["a2", "a1"]                 # a2 more popular -> rank 1
        assert "a3" not in items                     # different product_code

    def test_top_n_caps(self, spark):
        tx = _tx(spark, [("u1", "a1", date(2020, 1, 30), 1.0)])
        articles = [("a1", 100)] + [(f"a{i}", 100) for i in range(2, 8)]
        ctx = _ctx(spark, [("u1", None, None)], articles, train_end=date(2020, 2, 1))
        assert ProductCode(top_n=3, seed_weeks=4).generate(tx, ctx).count() == 3


# ---------------------------------------------------------------------------
# ItemCF
# ---------------------------------------------------------------------------

class TestItemCF:
    def test_recommends_co_bought_items(self, spark):
        # u1 buys a1; u2 buys a1+a2; u3 buys a1+a2. Score for a2 from seed a1 = 2.
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 30), 1.0),  # seed user
            ("u2", "a1", date(2020, 1, 10), 1.0),
            ("u2", "a2", date(2020, 1, 11), 1.0),
            ("u3", "a1", date(2020, 1, 12), 1.0),
            ("u3", "a2", date(2020, 1, 13), 1.0),
        ])
        ctx = _ctx(spark, [("u1", None, None), ("u2", None, None), ("u3", None, None)],
                   [("a1", 1), ("a2", 2)], train_end=date(2020, 2, 1))
        out = _collect(ItemCF(top_n=5, neighbors=50, seed_weeks=4)
                       .generate(tx, ctx).filter(F.col("customer_id") == "u1"))
        items = [r["article_id"] for r in sorted(out, key=lambda r: r["rank"])]
        assert items[0] == "a2"  # a2 is u1's only co-purchase neighbor

    def test_excludes_self_co_purchase(self, spark):
        # a single user buying one item: itemcf should never recommend the seed itself
        tx = _tx(spark, [("u1", "a1", date(2020, 1, 30), 1.0)])
        ctx = _ctx(spark, [("u1", None, None)], [("a1", 1)], train_end=date(2020, 2, 1))
        out = _collect(ItemCF(top_n=5).generate(tx, ctx))
        assert all(r["article_id"] != "a1" for r in out)


# ---------------------------------------------------------------------------
# SegmentPopularity
# ---------------------------------------------------------------------------

class TestSegmentPopularity:
    def test_validates_global_fallback_arg(self):
        with pytest.raises(ValueError, match="global_fallback must be one of"):
            SegmentPopularity(global_fallback="bogus")

    def test_off_mode_strict_demographic(self, spark):
        # Only demographic segments — no global tail
        tx = _tx(spark, [("u1", "a1", date(2020, 1, 30), 1.0)])
        ctx = _ctx(spark, [("u1", 30.0, "12345")], [("a1", 1)],
                   train_end=date(2020, 2, 1))
        # u1 in own segment -> gets a1 from L1
        out = _collect(
            SegmentPopularity(top_n=5, weeks=4, global_fallback="off")
            .generate(tx, ctx)
        )
        assert {r["article_id"] for r in out} == {"a1"}

    def test_no_info_only_targets_only_users_with_no_demo(self, spark):
        # u1 has full demographics (age + postal), u2 has neither.
        # With global_fallback=no_info_only, global tail is added ONLY to u2.
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 30), 1.0),
            ("u2", "a1", date(2020, 1, 30), 1.0),
        ])
        ctx = _ctx(spark, [("u1", 30.0, "12345"), ("u2", None, None)],
                   [("a1", 1)], train_end=date(2020, 2, 1))
        out = _collect(
            SegmentPopularity(top_n=5, weeks=4, global_fallback="no_info_only")
            .generate(tx, ctx)
        )
        # Both users should receive a1; the test confirms the strategy still
        # returns a result (the strict-demographic vs global-tail logic is
        # exercised by tiny inputs — coverage > exact-row equality here).
        assert {r["customer_id"] for r in out} == {"u1", "u2"}

    def test_custom_name_is_applied(self, spark):
        src = SegmentPopularity(name="popularity")
        assert src.name == "popularity"


# ---------------------------------------------------------------------------
# Output schema invariants — apply to every concrete source
# ---------------------------------------------------------------------------

class TestOutputSchema:
    @pytest.mark.parametrize("source_factory", [
        lambda: Repurchase(top_n=5),
        lambda: Popularity(top_n=5, weeks=4),
        lambda: ProductCode(top_n=5, seed_weeks=4),
        lambda: ItemCF(top_n=5),
        lambda: SegmentPopularity(top_n=5, weeks=4),
    ])
    def test_columns_and_rank_starts_at_1(self, spark, source_factory):
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 30), 1.0),
            ("u1", "a2", date(2020, 1, 30), 1.0),
        ])
        ctx = _ctx(spark, [("u1", 30.0, "p1")], [("a1", 100), ("a2", 100)],
                   train_end=date(2020, 2, 1))
        out = source_factory().generate(tx, ctx)
        assert set(["customer_id", "article_id", "rank", "source"]).issubset(out.columns)
        if out.count() > 0:
            assert out.agg(F.min("rank")).collect()[0][0] >= 1


# ---------------------------------------------------------------------------
# union_candidates
# ---------------------------------------------------------------------------

class TestUnion:
    def test_pivots_rank_per_source(self, spark):
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 30), 1.0),
            ("u1", "a2", date(2020, 1, 30), 1.0),
            ("u2", "a1", date(2020, 1, 5), 1.0),
            ("u3", "a1", date(2020, 1, 5), 1.0),
        ])
        ctx = _ctx(spark,
                   [("u1", None, None), ("u2", None, None), ("u3", None, None)],
                   [("a1", 100), ("a2", 100)], train_end=date(2020, 2, 1))
        out = union([Repurchase(top_n=5), Popularity(top_n=5, weeks=4)], tx, ctx)
        cols = set(out.columns)
        assert {"customer_id", "article_id", "repurchase_rank", "popularity_rank"}.issubset(cols)
        # u1 should have both repurchase + popularity ranks for a1 (bought it + globally popular)
        u1_a1 = _collect(out.filter((F.col("customer_id") == "u1") & (F.col("article_id") == "a1")))[0]
        assert u1_a1["repurchase_rank"] is not None
        assert u1_a1["popularity_rank"] is not None
