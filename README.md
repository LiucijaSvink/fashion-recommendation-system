# Personalised Fashion Recommendation System

A two-stage recommendation system that predicts the **12 products each customer is most likely to purchase in the following week**.

**31.8M transactions · 1.37M customers · 105K articles**

## Business problem

For an e-commerce retailer, the challenge is deciding **which products to show to each customer** from a large catalogue.

A bestseller strategy recommends popular products, but does not use an individual's purchase history. At the same time, evaluating every product for every customer becomes impractical at large scale.

This project explores a two-stage approach: first identify products that are plausible for a customer, then rank those products according to their likelihood of purchase.

## Approach

```text
Customer & product data
          │
          ▼
┌─────────────────────────────┐
│ Candidate Retrieval         │
│                             │
│ Repurchase                  │
│ ItemCF                      │
│ Product variants            │
│ Popularity                  │
│ Graph embeddings            │
└─────────────┬───────────────┘
              │
              ▼
       ~145 candidates
              │
              ▼
┌─────────────────────────────┐
│ Feature Engineering         │
│                             │
│ Customer behaviour          │
│ Product demand              │
│ Customer–product history    │
│ Retrieval signals           │
└─────────────┬───────────────┘
              │
              ▼
     LightGBM LambdaRank
              │
              ▼
       Top 12 products
```

Five retrieval methods generate approximately **145 candidates per customer**. A LightGBM LambdaRank model then ranks them using **53 customer, product, interaction, and retrieval features**.

The pipeline is implemented with **PySpark** for data processing.

## Results

Evaluation uses rolling time-based validation and a final **untouched holdout week**.

|                        | Recommendation model | Bestseller baseline |
| ---------------------- | -------------------: | ------------------: |
| **Hit rate @12**       |           **13.69%** |               4.97% |
| **Revenue captured**   |            **5.91%** |               1.81% |
| **Purchases captured** |            **5.41%** |               1.67% |
| **Catalogue coverage** |           **22.09%** |               0.01% |

The recommendation model identified at least one purchased product in the top 12 for **13.69% of purchasing customers**, compared with **4.97%** for the bestseller baseline.

On the Kaggle leaderboard for the same task, the model scores **0.02906 public / 0.02905 private** (MAP@12).

These are offline results; an online A/B test would be needed to measure incremental business impact.

## Business application

The same retrieval-and-ranking structure could be applied to e-commerce businesses with customer purchase and product data.

Possible applications include:

* homepage recommendations
* related-product recommendations
* personalised product feeds
* repeat-purchase recommendations
* personalised email campaigns

## Project structure

```text
notebook/
  pipeline.ipynb        End-to-end case study: retrieval, ranking, evaluation
  eda.ipynb             Exploratory analysis of purchase behaviour
configs/config.json     Configuration of the shipped model

src/recolib/            Project-agnostic recommender library
  config.py             Experiment configuration and schema mapping
  data/                 Loading, preprocessing, temporal splitting
  retrieval/            Candidate sources and their union
  features/             Feature sources and derived transforms
  modeling/             LightGBM LambdaRank ranker
  metrics/              MAP, NDCG, hit rate, precision, recall @k
  pipeline/             Stage orchestration and run caching
  analysis/             Fold, retrieval and commercial diagnostics
  backends/             Spark session management
  cli.py                Command-line entry point

eda/                    Analysis helpers for the EDA notebook
tests/                  Unit tests (214)
```

## How to run

Install dependencies and the library:

```bash
pip install -r requirements.txt
pip install -e .
```

Download the dataset (see below) and point the pipeline at it:

```bash
export HM_RAW_DIR=~/HM_dataset/hm_data   # the Kaggle CSVs
export HM_WORK_DIR=~/hm_data             # generated folds, features, models
```

Run the main pipeline from:

```text
notebook/pipeline.ipynb
```

The notebook covers candidate retrieval, candidate union, feature engineering, ranking, and evaluation.

The same run is available from the command line:

```bash
recolib run --config configs/config.json
```

Exploratory analysis is available in:

```text
notebook/eda.ipynb
```

It reads the prepared tables, so run the pipeline's first stage before it
(`recolib run --config configs/config.json --stages prepare`, or the
corresponding notebook cell).

## Dataset

This project uses the public **H&M Personalized Fashion Recommendations** dataset from
[Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data).

Download `transactions_train.csv`, `articles.csv`, `customers.csv` and
`sample_submission.csv` into `HM_RAW_DIR` — about 4 GB in total. The article
images are not used.

Be aware of the disk cost: candidates and features are materialised per fold,
so a full six-fold run writes roughly **20 GB per fold** (~140 GB) into
`HM_WORK_DIR`. To try the pipeline on less disk, reduce `n_folds` or set
`max_candidates` in `configs/config.json`.
