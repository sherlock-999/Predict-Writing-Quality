# Linking Writing Process to Writing Quality

Kaggle competition: predict a holistic essay score (0.5 – 6.0) from keystroke log data.

## Project Structure

```
writing_process/
├── data/                            # raw competition data (not tracked)
│   ├── train_logs.csv
│   ├── train_scores.csv
│   ├── test_logs.csv
│   ├── sample_submission.csv
│   ├── train_features_v4.csv        # precomputed features cache (not tracked)
│   └── test_features_v4.csv         # precomputed features cache (not tracked)
│
├── v4_features.py                   # feature engineering module (152 features)
├── precompute_features.py           # precompute + cache features for train & test
│
├── train_lgbm.py                    # LightGBM: 5-seed × 10-fold CV → v4_feature_lgbm/
├── train_xgb.py                     # XGBoost:  5-seed × 10-fold CV → xgb_model/
├── train_catboost.py                # CatBoost: 5-seed × 10-fold CV → catboost_model/
│
├── predict_lgbm_v4_features.py      # Kaggle inference — LightGBM only
├── predict_xgb.py                   # Kaggle inference — XGBoost only
├── predict_catboost.py              # Kaggle inference — CatBoost only
│
├── predict_lgbm_xgb_ensemble.py     # Kaggle inference — LGBM + XGBoost blend
├── predict_lgbm_catboost_ensemble.py# Kaggle inference — LGBM + CatBoost blend
├── predict_xgb_catboost_ensemble.py # Kaggle inference — XGBoost + CatBoost blend
├── predict_all3_ensemble.py         # Kaggle inference — all three models blended
│
├── v4_feature_lgbm/                 # saved LGBM fold models (not tracked)
├── xgb_model/                       # saved XGBoost fold models (not tracked)
├── catboost_model/                  # saved CatBoost fold models (not tracked)
│   └── tfidf/
│       └── tfidf_svd.pkl            # fitted TF-IDF SVD pipeline (shared by all models)
│
├── lgbm_plots/                      # output plots from train_lgbm.py
├── xgb_plots/                       # output plots from train_xgb.py
├── catboost_plots/                  # output plots from train_catboost.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
conda create -n exp python=3.11
conda activate exp
pip install -r requirements.txt
```

## Usage

Run in order:

```bash
# 1. Precompute features — caches train & test feature CSVs + saves tfidf_svd.pkl
conda run -n exp python precompute_features.py

# 2. Train models — each saves 50 fold models (5 seeds × 10 folds)
conda run -n exp python train_lgbm.py
conda run -n exp python train_xgb.py
conda run -n exp python train_catboost.py

# 3. Predict — upload the relevant predict_*.py to a Kaggle notebook
#    (all feature engineering is inlined; no local imports needed)
```

## Features — v4 (152 Total)

All features are computed per essay from the raw keystroke log.
Feature engineering is defined in `v4_features.py` and inlined in every predict script.

| # | Category | Count | Examples |
|---|---|---|---|
| 1 | **Word Characteristics** | 16 | word length (mean, max, quantiles), word count stats |
| 2 | **Sentence Characteristics** | 16 | sentence length, words per sentence, sentence count |
| 3 | **Paragraph Characteristics** | 17 | paragraph length, words per paragraph, paragraph count |
| 4 | **Punctuation & Formatting** | 11 | comma, period, newline, shift, caps-lock counts (per word) |
| 5 | **Typing Speed & Fluency** | 14 | keys per second, action time stats, key diversity |
| 6 | **Pausing Behavior** | 10 | idle gaps, pause counts at 0.5s / 1s / 1.5s / 2s / 3s bands |
| 7 | **Revision Behavior** | 7 | deletion counts, revision burst lengths |
| 8 | **Production Flow & Navigation** | 27 | production burst lengths, cursor movement, session timing |
| 9 | **Readability & Consistency** | 8 | ARI, Coleman-Liau, sentence/paragraph length std & CV, WPM |
| 10 | **Revision Timing** | 5 | fraction of deletions in early / mid / late thirds of session |
| 11 | **TF-IDF SVD** | 20 | char 2–4-gram TF-IDF → TruncatedSVD(20) on reconstructed text |

> **Note**: All alphabetic characters in `text_change` are anonymised as `q`.
> Punctuation (`.`, `,`, `?`, `'`, `-`, `\n`) and formatting keys are preserved.

### v4 vs v3 vs v2

v4 is built on the **v2 base** (119 features) with 33 additions (categories 9–11).
The v3 additions (`verbosity`, `idle_smallest_latency`, `initial_pause`, `ts_*` time-series features)
were excluded from v4 because they degraded OOF RMSE.

## Models

All three models use **5-seed × 10-fold stratified CV** (50 models each).
Stratification is on the exact essay score string so each fold is score-balanced.

| Model | OOF RMSE | Saved to | Model files |
|---|---|---|---|
| LightGBM | **0.6021** | `v4_feature_lgbm/` | `lgbm_s{seed}_fold{fold}.txt` |
| XGBoost | **0.6047** | `xgb_model/` | `xgb_s{seed}_fold{fold}.json` |
| CatBoost | **0.5983** | `catboost_model/` | `catboost_s{seed}_fold{fold}.cbm` |

### TF-IDF SVD Pipeline

The TF-IDF SVD transformer (`tfidf_svd.pkl`) is:
- Fitted **once** on the training corpus in `precompute_features.py`
- Saved to `{MODELS_DIR}/tfidf/tfidf_svd.pkl`
- Shared across all three model types at both train and inference time
- Uploaded to Kaggle as a separate dataset: `writing-quality-tfidf`

## Kaggle Submission

Each predict script is self-contained (all feature engineering inlined, no local imports).

### Dataset paths on Kaggle

| Dataset | Kaggle path |
|---|---|
| Competition data | `/kaggle/input/competitions/linking-writing-processes-to-writing-quality` |
| LightGBM models | `/kaggle/input/datasets/sherlocked999/v4-feature-lgbm/v4_feature_lgbm` |
| XGBoost models | `/kaggle/input/datasets/sherlocked999/xgb-model/xgb_model` |
| CatBoost models | `/kaggle/input/datasets/sherlocked999/catboost-model/catboost_model` |
| TF-IDF pipeline | `/kaggle/input/datasets/sherlocked999/writing-quality-tfidf/tfidf/tfidf_svd.pkl` |

### Ensemble blend weights (inverse-RMSE)

| Script | Models | LGBM | XGBoost | CatBoost |
|---|---|---|---|---|
| `predict_lgbm_xgb_ensemble.py` | LGBM + XGB | 0.501 | 0.499 | — |
| `predict_lgbm_catboost_ensemble.py` | LGBM + CB | 0.499 | — | 0.501 |
| `predict_xgb_catboost_ensemble.py` | XGB + CB | — | 0.497 | 0.503 |
| `predict_all3_ensemble.py` | All three | 0.331 | 0.329 | 0.340 |
