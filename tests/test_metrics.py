"""Unit tests for the pure-Python ranking metrics."""
from recolib.metrics import (
    apk, mapk, ndcg_at_k, precision_at_k, recall_at_k, mrr_at_k,
    hit_rate_at_k, hit_rate,
)


def test_apk_perfect_and_miss():
    assert apk({"a"}, ["a"], k=12) == 1.0
    assert apk({"a"}, ["x", "y"], k=12) == 0.0
    assert apk(set(), ["a"], k=12) == 0.0          # no relevant items -> 0


def test_apk_position_weighting():
    # one hit at rank 1 vs rank 3 (single relevant item)
    assert apk({"a"}, ["a", "x", "y"], k=12) == 1.0
    assert round(apk({"a"}, ["x", "y", "a"], k=12), 4) == round(1 / 3, 4)


def test_apk_two_hits():
    # hits at rank 1 and 3, 2 relevant items: (1/1 + 2/3)/2
    assert round(apk({"a", "b"}, ["a", "x", "b"], k=12), 4) == round((1 + 2 / 3) / 2, 4)


def test_apk_dedup_predictions():
    # duplicate predicted item should not double-count
    assert apk({"a"}, ["a", "a"], k=12) == 1.0


def test_mapk_over_population():
    gt = {"u1": {"a", "b"}, "u2": {"c"}}
    preds = {"u1": ["a", "x", "b"], "u2": ["y", "c"]}
    # u1 AP = (1 + 2/3)/2 = 5/6 ; u2 AP = 1/2 ; mean = (5/6 + 1/2)/2 = 8/12 = 2/3
    assert round(mapk(preds, gt, k=12), 4) == round(2 / 3, 4)
    # averaging over a larger population (a non-buyer) drags it down:
    # (5/6 + 1/2 + 0)/3 = (8/6)/3 = 8/18 = 4/9
    assert round(mapk(preds, gt, k=12, users=["u1", "u2", "u3"]), 4) == round(4 / 9, 4)


def test_ndcg_at_k_perfect_and_miss():
    # 1 relevant at rank 1: DCG = 1/log2(2) = 1; IDCG = 1; NDCG = 1
    assert ndcg_at_k({"u1": ["a"]}, {"u1": {"a"}}, k=12) == 1.0
    # No hit -> 0
    assert ndcg_at_k({"u1": ["x"]}, {"u1": {"a"}}, k=12) == 0.0


def test_ndcg_at_k_position_discount():
    # Hit at position 2 should give NDCG < 1
    val = ndcg_at_k({"u1": ["x", "a"]}, {"u1": {"a"}}, k=12)
    assert 0 < val < 1


def test_precision_at_k():
    # 1 hit out of 3 predictions -> 1/3
    val = precision_at_k({"u1": ["a", "x", "y"]}, {"u1": {"a"}}, k=3)
    assert round(val, 4) == round(1 / 3, 4)


def test_recall_at_k_default_pop_is_buyers():
    gt = {"u1": {"a", "b"}, "u2": {"c"}}
    preds = {"u1": ["a", "x"], "u2": ["c", "y"]}
    # u1 recall 1/2, u2 recall 1/1 -> mean 0.75
    assert recall_at_k(preds, gt, k=12) == 0.75


def test_recall_at_k_with_custom_users():
    # adding a non-buyer (u3) drags mean down
    gt = {"u1": {"a", "b"}, "u2": {"c"}}
    preds = {"u1": ["a", "x"], "u2": ["c", "y"]}
    # (1/2 + 1 + 0)/3 = 1/2
    assert recall_at_k(preds, gt, k=12, users=["u1", "u2", "u3"]) == 0.5


def test_mrr_at_k():
    # First hit at position 2 -> 1/2
    assert mrr_at_k({"u1": ["x", "a", "y"]}, {"u1": {"a"}}, k=12) == 0.5
    # No hit -> 0
    assert mrr_at_k({"u1": ["x", "y"]}, {"u1": {"a"}}, k=12) == 0.0


def test_hit_rate_at_k():
    # 2 of 3 users get at least one hit -> 2/3
    preds = {"u1": ["a"], "u2": ["x"], "u3": ["b"]}
    gt = {"u1": {"a"}, "u2": {"y"}, "u3": {"b"}}
    assert round(hit_rate_at_k(preds, gt, k=12), 4) == round(2 / 3, 4)


def test_hit_rate_set_based():
    cand = {("u1", "a"), ("u1", "z"), ("u2", "c")}
    truth = {("u1", "a"), ("u1", "b"), ("u2", "c")}
    assert round(hit_rate(cand, truth), 4) == round(2 / 3, 4)
    assert hit_rate(cand, set()) == 0.0
