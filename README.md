# Linking Writing Process to Writing Quality

Kaggle competition: predict a holistic essay score (0.5 – 6.0) from raw keystroke log data only — no essay text is directly provided.

## Overview

The input is a sequence of keystroke events per essay (key pressed, timestamp, cursor position, word count at that moment). The goal is to infer writing quality purely from the *process* of writing, not from the finished product.

Our approach: engineer 226 features that quantify writing behaviour across 14 categories, then train three gradient-boosted tree models (LightGBM, XGBoost, CatBoost) using repeated stratified cross-validation, and blend predictions at inference time.

For full details see:
- [FEATURES.md](FEATURES.md) — what each feature captures and why
- [MODELS.md](MODELS.md) — training setup, CV strategy, and ensembling
- [RESULTS.md](RESULTS.md) — leaderboard scores, OOF analysis, and conclusions

## Project Structure

```
writing_process/
├── data/                               # raw competition data (not tracked)
│   ├── train_logs.csv
│   ├── train_scores.csv
│   ├── test_logs.csv
│   ├── sample_submission.csv
│   ├── train_features_v6.csv           # precomputed feature cache (not tracked)
│   └── test_features_v6.csv            # precomputed feature cache (not tracked)
│
├── v6_features.py                      # feature engineering module (226 features)
├── precompute_features_v6.py           # precompute + cache features for train & test
│
├── train_lgbm.py                       # LightGBM: 5-seed × 10-fold CV → v6_features_lgbm/
├── train_xgb.py                        # XGBoost:  5-seed × 10-fold CV → xgb_model/
├── train_catboost.py                   # CatBoost: 5-seed × 10-fold CV → catboost_model/
│
├── predict_lgbm_v6_features.py         # Kaggle inference — LightGBM only
├── predict_xgb.py                      # Kaggle inference — XGBoost only
├── predict_catboost.py                 # Kaggle inference — CatBoost only
│
├── predict_lgbm_xgb_ensemble.py        # Kaggle inference — LGBM + XGBoost blend
├── predict_lgbm_catboost_ensemble.py   # Kaggle inference — LGBM + CatBoost blend
├── predict_xgb_catboost_ensemble.py    # Kaggle inference — XGBoost + CatBoost blend
├── predict_all3_ensemble.py            # Kaggle inference — all three models blended
│
├── oof_correlation.py                  # recompute OOF preds, plot correlation & blend weights
│
├── v6_features_lgbm/                   # saved LightGBM fold models (not tracked)
├── xgb_model/                          # saved XGBoost fold models (not tracked)
├── catboost_model/                     # saved CatBoost fold models (not tracked)
│   └── tfidf_v6/
│       ├── tfidf_svd.pkl               # fitted text TF-IDF SVD pipeline
│       └── event_tfidf_svd.pkl         # fitted event TF-IDF SVD pipeline
│
├── lgbm_plots/                         # output plots from train_lgbm.py
├── xgb_plots/                          # output plots from train_xgb.py
├── catboost_plots/                     # output plots from train_catboost.py
├── oof_correlation_plots/              # output plots from oof_correlation.py
│
├── FEATURES.md
├── MODELS.md
├── RESULTS.md
├── requirements.txt
└── README.md
```

## Setup

```bash
conda create -n exp python=3.11
conda activate exp
pip install -r requirements.txt
```

## Reproducing the Pipeline

```bash
# 1. Precompute features — caches train & test feature CSVs + saves both TF-IDF pipelines
conda run -n exp python precompute_features_v6.py

# 2. Train all three models (each saves 50 fold models: 5 seeds × 10 folds)
conda run -n exp python train_lgbm.py
conda run -n exp python train_xgb.py
conda run -n exp python train_catboost.py

# 3. (Optional) Analyse OOF correlations and optimal blend weights
conda run -n exp python oof_correlation.py

# 4. Predict — upload the relevant predict_*.py to a Kaggle notebook
#    All feature engineering is inlined; no local imports needed.
```

## Kaggle Dataset Paths

| Asset | Kaggle path |
|---|---|
| Competition data | `/kaggle/input/competitions/linking-writing-processes-to-writing-quality` |
| LightGBM models | `/kaggle/input/datasets/sherlocked999/v6-features-lgbm/v6_features_lgbm` |
| XGBoost models | `/kaggle/input/datasets/sherlocked999/xgb-model/xgb_model` |
| CatBoost models | `/kaggle/input/datasets/sherlocked999/catboost-model/catboost_model` |
| TF-IDF pipelines | `/kaggle/input/datasets/sherlocked999/tfidf-more/tfidf_v6/` |
