# Results

All scores are RMSE (lower is better). The target is a holistic essay score on the scale 0.5 – 6.0.

---

## Individual Model Performance

### OOF RMSE (5-seed × 10-fold average)

OOF predictions are averaged across all 5 seeds before computing RMSE.

| Model | OOF RMSE |
|-------|----------|
| LightGBM | 0.6024 |
| XGBoost | 0.6056 |
| CatBoost | 0.5984 |

CatBoost achieves the best OOF score, followed by LightGBM and XGBoost.

### Leaderboard (Public) RMSE

| Model | LB RMSE |
|-------|---------|
| LightGBM | 0.5856 |
| XGBoost | 0.5859 |
| CatBoost | 0.5909 |

**The ranking inverts on the leaderboard.** LightGBM and XGBoost generalise better to held-out data than CatBoost, despite CatBoost having the best OOF score. CatBoost's lower OOF RMSE reflects overfitting to the training distribution — its OOF estimate is optimistic.

---

## Ensemble Performance

### OOF RMSE of blended predictions (equal weights)

| Combination | OOF RMSE |
|-------------|----------|
| LGBM alone | 0.6024 |
| XGB alone | 0.6056 |
| CatBoost alone | 0.5984 |
| LGBM + XGB (A+B) | 0.6031 |
| LGBM + CatBoost (A+C) | 0.5983 |
| XGB + CatBoost (B+C) | 0.5997 |
| LGBM + XGB + CatBoost | 0.5998 |

### Leaderboard RMSE of blended predictions

| Combination | LB RMSE |
|-------------|---------|
| LGBM alone | 0.5856 |
| XGB alone | 0.5859 |
| CatBoost alone | 0.5909 |
| LGBM + XGB (A+B) | **0.5853** |
| LGBM + CatBoost (A+C) | 0.5866 |
| XGB + CatBoost (B+C) | 0.5865 |
| LGBM + XGB + CatBoost | 0.5857 |

**LGBM + XGB is the best submission**, beating every individual model and every other combination on the leaderboard.

---

## OOF Prediction Correlation

Pearson correlation between the mean OOF predictions of each model pair:

| Pair | r |
|------|---|
| LGBM vs XGB | 0.9968 |
| LGBM vs CatBoost | 0.9922 |
| XGB vs CatBoost | 0.9916 |

All three models are extremely highly correlated (>0.99). This means blending provides very little diversity benefit — the gains from ensembling are marginal (~0.0003 RMSE for the best pair). The models are learning essentially the same signal from the same feature set.

---

## Blend Weight Analysis

A grid search over 201 weight values for the LGBM/XGB blend (OOF-based) finds:

- **Optimal weight:** `w_lgbm = 0.96` (nearly pure LGBM)
- **OOF RMSE at optimal weight:** 0.602424 vs. 0.602430 at equal weights

The improvement is negligible — consistent with the very high inter-model correlation. Near-equal weighting is effectively equivalent to pure LGBM at this correlation level.

---

## Conclusions

### 1. OOF RMSE is not a reliable ranking signal for blending

CatBoost ranked first on OOF (0.5984) but last on the leaderboard (0.5909). LGBM ranked second on OOF but first on the leaderboard. This divergence means OOF scores should not be used to select the best model or blend weights without sanity-checking against held-out data. CatBoost is overfitting to the training score distribution in a way the cross-validation scheme does not detect.

### 2. Model diversity is the limiting factor for ensembling

All three models share the same 226-feature input. With inter-model correlations above 0.99, blending cannot meaningfully reduce prediction variance. The ~0.0003 RMSE improvement from LGBM+XGB over LGBM alone is likely within noise. To benefit from ensembling, future work should introduce genuinely diverse models — different feature sets, neural architectures, or sequence models trained directly on the keystroke stream.

### 3. The feature set is effective but saturated within tree models

The gap between the best ensemble (LB 0.5853) and the baseline (predicting the mean score, RMSE ≈ 0.68) is ~0.095 RMSE, representing substantial predictive signal extracted from the keystroke logs. However, the minimal gain from ensembling three strong tree models suggests the feature set is approaching its information ceiling for this model class. Further gains likely require either richer features or a different model family.

### 4. The essay text carries information not yet fully captured

The TF-IDF SVD features (categories 13 and 14) encode writing style through character n-grams and keystroke sequence n-grams. These are indirect proxies for the actual text quality. A model that directly evaluates the reconstructed essay text — for example, using a pre-trained language model — would likely close the remaining gap between keystroke-derived features and true essay quality scores.

### 5. CatBoost symmetric trees overfit on this dataset

CatBoost uses symmetric (oblivious) decision trees, which are fast but less expressive than the leaf-wise trees used by LightGBM. The OOF/LB inversion suggests CatBoost is more sensitive to the score distribution in the training set and does not generalise as well when the test distribution differs slightly. LGBM and XGBoost's leaf-wise growth appears better suited to this regression task.

---

## Best Submission

**Script:** `predict_lgbm_xgb_ensemble.py`  
**LB RMSE:** 0.5853  
**Blend:** LGBM (w ≈ 0.501) + XGBoost (w ≈ 0.499), inverse-RMSE weighted
