"""recolib — a small two-stage (retrieval + learning-to-rank) recommender library.

    import recolib as rl
    schema = rl.Schema()                                       # H&M defaults
    context    = rl.candidates.RetrievalContext(spark, customers, articles, train_end,
                                            schema=schema)
    cands  = rl.candidates.union([rl.candidates.Repurchase(80),
                                  rl.candidates.SegmentPopularity(50)], transactions, context)
    fctx   = rl.features.FeatureContext(spark, train_end, transactions, history,
                                        customers, articles, schema=schema)
    sources = [rl.features.UserAggregates(), rl.features.ItemAggregates(),
               rl.features.CategoryAffinity("product_code"),
               rl.features.UserItemHistory()]
    feats  = cands
    for src in sources:
        block, keys = src.compute(fctx)
        feats = feats.join(block, on=keys, how="left")
    feats = rl.features.derived_ratios(rl.features.attach_labels(feats, label_df, schema))
    model  = rl.reranker.LambdaRanker(schema=schema).fit(train_pdf)
    print(rl.metrics.mapk(model.recommend(eval_pdf), ground_truth, k=12))
"""
from . import analysis, backends, data, features, modeling, retrieval
from .schema import DEFAULT_SCHEMA, Schema
from . import metrics

__version__ = "0.1.0"
__all__ = ["data", "retrieval", "features", "modeling", "backends", "metrics", "analysis",
           "Schema", "DEFAULT_SCHEMA"]
