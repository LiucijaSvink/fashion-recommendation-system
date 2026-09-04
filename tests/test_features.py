"""Tests for recolib.features — FeatureSource subclasses + transforms + ABC contract."""
from __future__ import annotations

from datetime import date

import pytest
from pyspark.sql import functions as F

from recolib.features import (
    ArticleMetadata,
    CategoryAffinity,
    CustomerMetadata,
    FeatureContext,
    FeatureSource,
    ItemAggregates,
    ItemBuyerDemographic,
    ItemCFScore,
    UserAggregates,
    UserItemHistory,
    attach_labels,
    derived_ratios,
    source_flags,
)
from recolib.schema import Schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TX_SCHEMA = "customer_id string, article_id string, t_dat date, price double, sales_channel_id int"
_CUST_SCHEMA = "customer_id string, age double, postal_code string, club_member_status string, fashion_news_frequency string"
_ART_SCHEMA = (
    "article_id string, product_code long, product_type_no long, "
    "department_no long, product_group_name string, "
    "garment_group_no long, section_no long, index_group_no long, colour_group_code long"
)


def _tx(spark, rows):
    return spark.createDataFrame(
        [tuple(list(r) + [1] * (5 - len(r))) for r in rows], schema=_TX_SCHEMA,
    )


def _customers(spark, rows):
    return spark.createDataFrame(rows, schema=_CUST_SCHEMA)


def _articles(spark, rows):
    return spark.createDataFrame(rows, schema=_ART_SCHEMA)


def _fctx(spark, transactions, history, customers, articles, end=date(2020, 2, 1), neighbors=None):
    return FeatureContext(
        spark=spark, end=end,
        transactions=transactions, history=history,
        customers=customers, articles=articles, neighbors=neighbors,
    )


def _collect(df):
    return [r.asDict() for r in df.collect()]


# Sample article rows that satisfy the article schema
def _art_row(article_id, product_code=100, product_type_no=10, department_no=1,
             product_group_name="g", garment_group_no=1, section_no=1,
             index_group_no=1, colour_group_code=1):
    return (article_id, product_code, product_type_no, department_no,
            product_group_name, garment_group_no, section_no,
            index_group_no, colour_group_code)


# ---------------------------------------------------------------------------
# ABC + Context
# ---------------------------------------------------------------------------

class TestBase:
    def test_abc_enforces_compute(self):
        class Broken(FeatureSource):
            name = "broken"
            produces = []
        with pytest.raises(TypeError, match="abstract"):
            Broken()

    def test_context_defaults(self, spark):
        ctx = _fctx(spark, _tx(spark, []), _tx(spark, []),
                    _customers(spark, []), _articles(spark, []))
        assert ctx.neighbors is None
        assert isinstance(ctx.schema, Schema)


# ---------------------------------------------------------------------------
# UserAggregates
# ---------------------------------------------------------------------------

class TestUserAggregates:
    def test_active_and_inactive_split(self, spark):
        # u1 active (recent); u2 inactive (only old purchases)
        history = _tx(spark, [
            ("u1", "a1", date(2020, 1, 28), 1.0),     # recent
            ("u2", "a1", date(2019, 6, 1), 1.0),      # old only
        ])
        ctx = _fctx(spark, _tx(spark, []), history,
                    _customers(spark, []), _articles(spark, []),
                    end=date(2020, 2, 1))
        block, keys = UserAggregates(recent_weeks=12).compute(ctx)
        assert keys == ["customer_id"]
        assert set(UserAggregates.produces).issubset(set(block.columns))
        rows = {r["customer_id"]: r for r in _collect(block)}
        assert rows["u1"]["is_active"] == 1
        assert rows["u2"]["is_active"] == 0

    def test_declared_columns_match_output(self, spark):
        history = _tx(spark, [("u1", "a1", date(2020, 1, 28), 1.0)])
        ctx = _fctx(spark, _tx(spark, []), history,
                    _customers(spark, []), _articles(spark, []))
        block, _ = UserAggregates().compute(ctx)
        for col in UserAggregates.produces:
            assert col in block.columns, f"declared `{col}` missing from output"


# ---------------------------------------------------------------------------
# ItemAggregates
# ---------------------------------------------------------------------------

class TestItemAggregates:
    def test_windows_are_parameterized(self, spark):
        src = ItemAggregates(windows=(1, 2))
        assert "item_purchases_1w" in src.produces
        assert "item_purchases_2w" in src.produces
        assert "item_purchases_12w" not in src.produces

    def test_window_filter_correct(self, spark):
        # End date 2020-02-01. 1w window starts 2020-01-25.
        tx = _tx(spark, [
            ("u1", "a1", date(2020, 1, 26), 1.0),    # in 1w
            ("u2", "a1", date(2020, 1, 26), 1.0),    # in 1w
            ("u3", "a1", date(2020, 1, 1),  1.0),    # outside 1w
        ])
        ctx = _fctx(spark, tx, _tx(spark, []), _customers(spark, []),
                    _articles(spark, []), end=date(2020, 2, 1))
        block, keys = ItemAggregates(windows=(1,)).compute(ctx)
        assert keys == ["article_id"]
        row = _collect(block.filter(F.col("article_id") == "a1"))[0]
        assert row["item_purchases_1w"] == 2
        assert row["item_purchases_all"] == 3
        assert row["item_repurchase_rate"] == 1.0    # 3 purchases / 3 unique buyers


# ---------------------------------------------------------------------------
# CategoryAffinity
# ---------------------------------------------------------------------------

class TestCategoryAffinity:
    def test_name_and_produces_reflect_key(self):
        src = CategoryAffinity("colour_group_code")
        assert src.name == "category_affinity_colour_group_code"
        assert src.produces == ["user_colour_group_code_purchases"]

    def test_out_name_overrides_suffix(self):
        src = CategoryAffinity("product_code", out_name="pcode")
        assert src.name == "category_affinity_pcode"
        assert src.produces == ["user_pcode_purchases"]

    def test_counts_purchases_per_category(self, spark):
        history = _tx(spark, [
            ("u1", "a1", date(2020, 1, 28), 1.0),    # product_code 100
            ("u1", "a2", date(2020, 1, 28), 1.0),    # product_code 100 again
            ("u1", "a3", date(2020, 1, 28), 1.0),    # product_code 200
        ])
        articles = _articles(spark, [
            _art_row("a1", product_code=100),
            _art_row("a2", product_code=100),
            _art_row("a3", product_code=200),
        ])
        ctx = _fctx(spark, _tx(spark, []), history,
                    _customers(spark, []), articles, end=date(2020, 2, 1))
        block, keys = CategoryAffinity("product_code").compute(ctx)
        assert keys == ["customer_id", "product_code"]
        rows = {(r["customer_id"], r["product_code"]): r["user_product_code_purchases"]
                for r in _collect(block)}
        assert rows[("u1", 100)] == 2
        assert rows[("u1", 200)] == 1


# ---------------------------------------------------------------------------
# UserItemHistory
# ---------------------------------------------------------------------------

class TestUserItemHistory:
    def test_aggregates_count_and_recency(self, spark):
        history = _tx(spark, [
            ("u1", "a1", date(2020, 1, 1),  1.0),
            ("u1", "a1", date(2020, 1, 20), 1.0),
        ])
        ctx = _fctx(spark, _tx(spark, []), history, _customers(spark, []),
                    _articles(spark, []), end=date(2020, 2, 1))
        block, keys = UserItemHistory().compute(ctx)
        assert set(keys) == {"customer_id", "article_id"}
        row = _collect(block.filter((F.col("customer_id") == "u1")
                                    & (F.col("article_id") == "a1")))[0]
        assert row["user_item_purchases"] == 2
        assert row["user_item_days_since_last"] == 12    # 2020-02-01 - 2020-01-20


# ---------------------------------------------------------------------------
# ItemBuyerDemographic
# ---------------------------------------------------------------------------

class TestItemBuyerDemographic:
    def test_mean_buyer_age(self, spark):
        history = _tx(spark, [
            ("u1", "a1", date(2020, 1, 1), 1.0),
            ("u2", "a1", date(2020, 1, 2), 1.0),
        ])
        customers = _customers(spark, [
            ("u1", 20.0, "p", "ACTIVE", "Regularly"),
            ("u2", 40.0, "p", "ACTIVE", "Regularly"),
        ])
        ctx = _fctx(spark, _tx(spark, []), history, customers,
                    _articles(spark, []))
        block, keys = ItemBuyerDemographic().compute(ctx)
        assert keys == ["article_id"]
        row = _collect(block.filter(F.col("article_id") == "a1"))[0]
        assert row["item_mean_buyer_age"] == 30.0


# ---------------------------------------------------------------------------
# CustomerMetadata
# ---------------------------------------------------------------------------

class TestCustomerMetadata:
    def test_indexes_categoricals_and_keeps_age(self, spark):
        customers = _customers(spark, [
            ("u1", 25.0, "p1", "ACTIVE",     "Regularly"),
            ("u2", 35.0, "p2", "PRE-CREATE", "NONE"),
        ])
        ctx = _fctx(spark, _tx(spark, []), _tx(spark, []), customers,
                    _articles(spark, []))
        block, keys = CustomerMetadata().compute(ctx)
        assert keys == ["customer_id"]
        cols = set(block.columns)
        assert {"age", "club_member_status_idx", "fashion_news_frequency_idx"}.issubset(cols)
        # Original string columns are dropped
        assert "club_member_status" not in cols


# ---------------------------------------------------------------------------
# ArticleMetadata
# ---------------------------------------------------------------------------

class TestArticleMetadata:
    def test_casts_codes_to_double(self, spark):
        articles = _articles(spark, [_art_row("a1", colour_group_code=5)])
        ctx = _fctx(spark, _tx(spark, []), _tx(spark, []), _customers(spark, []),
                    articles)
        block, keys = ArticleMetadata().compute(ctx)
        assert keys == ["article_id"]
        types = dict(block.dtypes)
        assert types["colour_group_code"] == "double"

    def test_codes_parameter_subset(self, spark):
        articles = _articles(spark, [_art_row("a1")])
        ctx = _fctx(spark, _tx(spark, []), _tx(spark, []), _customers(spark, []),
                    articles)
        src = ArticleMetadata(codes=("colour_group_code", "section_no"))
        block, _ = src.compute(ctx)
        assert set(block.columns) == {"article_id", "colour_group_code", "section_no"}


# ---------------------------------------------------------------------------
# ItemCFScore
# ---------------------------------------------------------------------------

class TestItemCFScore:
    def test_raises_without_neighbors(self, spark):
        ctx = _fctx(spark, _tx(spark, []), _tx(spark, []), _customers(spark, []),
                    _articles(spark, []))
        with pytest.raises(ValueError, match="neighbors"):
            ItemCFScore().compute(ctx)

    def test_scores_from_neighbors(self, spark):
        tx = _tx(spark, [("u1", "seedA", date(2020, 1, 28), 1.0)])
        neighbors = spark.createDataFrame(
            [("seedA", "a1", 5), ("seedA", "a2", 2)],
            "seed string, article_id string, cocount long",
        )
        ctx = _fctx(spark, tx, _tx(spark, []), _customers(spark, []),
                    _articles(spark, []), end=date(2020, 2, 1), neighbors=neighbors)
        block, keys = ItemCFScore(seed_weeks=4).compute(ctx)
        assert set(keys) == {"customer_id", "article_id"}
        rows = {(r["customer_id"], r["article_id"]): r["itemcf_score"]
                for r in _collect(block)}
        assert rows[("u1", "a1")] == 5
        assert rows[("u1", "a2")] == 2


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestTransforms:
    def test_source_flags_and_count(self, spark):
        df = spark.createDataFrame(
            [("u1", "a1", 1,    None), ("u1", "a2", None, 1)],
            "customer_id string, article_id string, sA_rank long, sB_rank long",
        )
        out = source_flags(df, ["sA", "sB"]).orderBy("article_id")
        rows = _collect(out)
        assert rows[0]["src_sA"] == 1 and rows[0]["src_sB"] == 0 and rows[0]["n_sources"] == 1
        assert rows[1]["src_sA"] == 0 and rows[1]["src_sB"] == 1 and rows[1]["n_sources"] == 1

    def test_attach_labels_none_fills_zero(self, spark):
        df = spark.createDataFrame([("u1", "a1")], "customer_id string, article_id string")
        out = attach_labels(df, None, Schema())
        assert _collect(out)[0]["label"] == 0

    def test_attach_labels_marks_positives(self, spark):
        df = spark.createDataFrame(
            [("u1", "a1"), ("u1", "a2")],
            "customer_id string, article_id string",
        )
        labels = spark.createDataFrame(
            [("u1", "a1")], "customer_id string, article_id string",
        )
        out = attach_labels(df, labels, Schema()).orderBy("article_id")
        rows = _collect(out)
        assert rows[0]["label"] == 1     # a1 is positive
        assert rows[1]["label"] == 0     # a2 not in labels

    def test_derived_ratios_computes_expected_cols(self, spark):
        df = spark.createDataFrame([
            (1.0, 2.0, 3, 6, 2, 5, 30.0, 25.0),
        ], "user_avg_price double, item_avg_price double, "
           "user_dept_purchases long, user_purchases long, "
           "user_item_purchases long, item_purchases_all long, "
           "age double, item_mean_buyer_age double")
        out = derived_ratios(df).collect()[0]
        assert out["price_ratio"] == pytest.approx(2.0, rel=1e-3)        # 2.0 / 1.0
        assert out["user_dept_ratio"] == pytest.approx(0.5, rel=1e-3)    # 3 / 6
        assert out["bought_before"] == 1                                  # 2 > 0
        assert out["age_gap"] == pytest.approx(5.0)                      # 30 - 25
        assert out["user_item_ratio"] == pytest.approx(2 / 6, rel=1e-3)


# ---------------------------------------------------------------------------
# Output-schema invariants
# ---------------------------------------------------------------------------

class TestOutputSchema:
    """Every source's `produces` declaration should match its compute output."""

    def _ctx_with_data(self, spark):
        tx = _tx(spark, [("u1", "a1", date(2020, 1, 28), 1.0)])
        history = _tx(spark, [
            ("u1", "a1", date(2020, 1, 28), 1.0),
            ("u2", "a1", date(2020, 1, 28), 1.0),
        ])
        customers = _customers(spark, [
            ("u1", 25.0, "p", "ACTIVE", "Regularly"),
            ("u2", 30.0, "p", "ACTIVE", "Regularly"),
        ])
        articles = _articles(spark, [_art_row("a1")])
        return _fctx(spark, tx, history, customers, articles)

    @pytest.mark.parametrize("source_factory", [
        lambda: UserAggregates(),
        lambda: ItemAggregates(windows=(1, 4)),
        lambda: UserItemHistory(),
        lambda: ItemBuyerDemographic(),
        lambda: CustomerMetadata(),
        lambda: ArticleMetadata(),
        lambda: CategoryAffinity("product_code"),
    ])
    def test_produces_columns_present_in_output(self, spark, source_factory):
        src = source_factory()
        block, _ = src.compute(self._ctx_with_data(spark))
        for col in src.produces:
            assert col in block.columns, f"{src.name} missing declared column {col}"
