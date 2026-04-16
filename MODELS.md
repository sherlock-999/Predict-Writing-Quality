# Models

## Overview

Three gradient-boosted tree models are trained on the same 226-feature set: **LightGBM**, **XGBoost**, and **CatBoost**. Each uses a 5-seed × 10-fold repeated stratified cross-validation scheme, producing 50 saved models per algorithm (150 total). At inference time, predictions are averaged within each family and optionally blended across families.

---

## Cross-Validation Strategy

### Repeated Stratified K-Fold

```
5 seeds × 10 folds = 50 models per algorithm
Seeds: [42, 21, 2022, 7, 4]
```

**Stratification** is on the exact essay score string (e.g., `"3.5"`), ensuring every fold contains the same proportion of each score level. This is important because the score distribution is imbalanced — scores cluster around 3.0–4.0 and the tails (1.0, 5.5, 6.0) are rare.

**Repeating across 5 seeds** reduces the variance of the out-of-fold (OOF) estimate. Different random seeds produce different fold splits, so averaging OOF predictions across seeds gives a more stable estimate of generalisation performance than a single 10-fold run.

**Why 10 folds?** With ~2,500 training essays, 10-fold CV means each validation set has ~250 essays — enough for a reliable RMSE estimate per fold, while still training on 90% of the data. Fewer folds (e.g. 5) would give noisier per-fold estimates; more folds would increase training time without meaningful benefit at this dataset size.

### Early Stopping

All three models use early stopping against the validation fold, preventing overfitting without manual tuning of `n_estimators`:

| Model | Max iterations | Early stopping rounds |
|-------|---------------|----------------------|
| LightGBM | — (callback-based) | 200 |
| XGBoost | 2000 | 50 |
| CatBoost | 2000 | 50 |

The true optimum number of trees is found automatically per fold.

---

## Model Configurations

### LightGBM

```python
{
    'num_leaves':        31,
    'min_child_samples': 20,
    'learning_rate':     0.05,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      1,
    'lambda_l1':         0.1,
    'lambda_l2':         0.1,
}
```

Saved as `lgbm_s{seed}_fold{fold}.txt`. Loaded at inference with `lgb.Booster(model_file=path)`.

### XGBoost

```python
{
    'learning_rate':    0.005,
    'objective':        'reg:squarederror',
    'reg_alpha':        0.000877,
    'reg_lambda':       2.543,
    'colsample_bynode': 0.784,
    'subsample':        0.899,
    'eta':              0.047,
    'tree_method':      'gpu_hist',
    'device':           'cuda',
    'n_estimators':     2000,
    'early_stopping_rounds': 50,
}
```

Parameters were tuned with Optuna. Saved as `xgb_s{seed}_fold{fold}.json`. Loaded at inference with `xgb.Booster(); booster.load_model(path)`.

### CatBoost

```python
{
    'loss_function':        'RMSE',
    'iterations':           2000,
    'learning_rate':        0.05,
    'depth':                6,
    'l2_leaf_reg':          3.0,
    'bootstrap_type':       'Bernoulli',
    'subsample':            0.8,
    'colsample_bylevel':    0.8,
    'min_data_in_leaf':     20,
    'early_stopping_rounds': 50,
}
```

Saved as `catboost_s{seed}_fold{fold}.cbm`. Loaded at inference with `CatBoostRegressor(); model.load_model(path)`.

---

## Inference

Each predict script is **self-contained**: all feature engineering functions are inlined, and no local module imports are required. This makes them drop-in compatible with Kaggle notebooks where the local repository is not available.

### Pipeline at inference time

```
test_logs.csv
      │
      ▼
  load tfidf_svd.pkl          ← char 2–4-gram text TF-IDF SVD pipeline
  load event_tfidf_svd.pkl    ← word 1–5-gram event TF-IDF SVD pipeline
      │
      ▼
  compute_features(logs, tfidf_pipeline, event_tfidf_pipeline)
      │   replays all keystroke edits, computes 226 features per essay
      │   applies both TF-IDF SVD transforms (transform only, no refit)
      ▼
  X_test  [n_essays × 226]
      │
      ▼
  load 50 fold models  (lgbm_s*_fold*.txt / xgb_s*_fold*.json / catboost_s*_fold*.cbm)
      │
      ▼
  average predictions across 50 folds → family-level prediction
      │
      ▼
  blend families → final score (clipped to [0.5, 6.0])
```

---

## Ensembling

### Motivation

No single model type is universally best. Tree-based models with different regularisation strategies (leaf-wise growth in LightGBM, level-wise in XGBoost, symmetric trees in CatBoost) can capture different aspects of the feature space. Blending their predictions reduces variance and smooths over individual model errors.

### Blending method: inverse-RMSE weighting

Each family's prediction is a simple average over its 50 fold models. The three family predictions are combined as a weighted average, where the weight is proportional to `1 / OOF_RMSE`:

```
weight_i = 1 / RMSE_i
final_pred = (w_lgbm × pred_lgbm + w_xgb × pred_xgb + w_catboost × pred_catboost)
             / (w_lgbm + w_xgb + w_catboost)
```

This gives higher weight to models with lower OOF error, without requiring a held-out validation set for meta-learning. It is equivalent to a simple linear blend optimised under the assumption that OOF RMSE is a reliable ranking signal.

### Available ensemble scripts

| Script | Models included | Blend weights (approx.) |
|--------|----------------|------------------------|
| `predict_lgbm_v6_features.py` | LGBM only | — |
| `predict_xgb.py` | XGB only | — |
| `predict_catboost.py` | CatBoost only | — |
| `predict_lgbm_xgb_ensemble.py` | LGBM + XGB | LGBM 0.501 / XGB 0.499 |
| `predict_lgbm_catboost_ensemble.py` | LGBM + CatBoost | LGBM 0.499 / CB 0.501 |
| `predict_xgb_catboost_ensemble.py` | XGB + CatBoost | XGB 0.497 / CB 0.503 |
| `predict_all3_ensemble.py` | All three | LGBM 0.333 / XGB 0.331 / CB 0.336 |

### OOF-based blend weight optimisation

`oof_correlation.py` performs a finer grid search (201 points) over the LGBM/XGB blend weight using OOF predictions directly. The optimal weight found is `w_lgbm = 0.96`, yielding a marginal OOF RMSE improvement over equal weighting — consistent with the OOF result showing LGBM slightly outperforming XGB.

---

## Saved Model Locations

```
v6_features_lgbm/
├── lgbm_s42_fold1.txt  …  lgbm_s4_fold10.txt    (50 models)
└── tfidf_v6/
    ├── tfidf_svd.pkl
    └── event_tfidf_svd.pkl

xgb_model/
└── xgb_s42_fold1.json  …  xgb_s4_fold10.json    (50 models)

catboost_model/
└── catboost_s42_fold1.cbm  …  catboost_s4_fold10.cbm  (50 models)
```
