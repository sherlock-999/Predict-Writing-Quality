# Results

All scores are RMSE (lower is better). The target is a holistic essay score on the scale 0.5 – 6.0.

---

## Score Summary

| Combination | CV (OOF) | Public LB | Private LB |
|-------------|----------|-----------|------------|
| LGBM alone | 0.6024 | 0.5856 | 0.5676 |
| XGB alone | 0.6056 | 0.5859 | 0.5688 |
| CatBoost alone | 0.5984 | 0.5909 | **0.5635** |
| LGBM + XGB | 0.6031 | **0.5853** | 0.5677 |
| LGBM + CatBoost | 0.5983 | 0.5866 | 0.5637 |
| XGB + CatBoost | 0.5997 | 0.5865 | 0.5641 |
| LGBM + XGB + CatBoost | 0.5998 | 0.5857 | 0.5647 |

---

## Individual Model Performance

### OOF RMSE (5-seed × 10-fold average)

OOF predictions are averaged across all 5 seeds before computing RMSE.

| Model | OOF RMSE |
|-------|----------|
| LightGBM | 0.6024 |
| XGBoost | 0.6056 |
| **CatBoost** | **0.5984** |

CatBoost achieves the best OOF score, followed by LightGBM and XGBoost.

### Public Leaderboard RMSE

| Model | Public LB |
|-------|-----------|
| **LightGBM** | **0.5856** |
| XGBoost | 0.5859 |
| CatBoost | 0.5909 |

The ranking inverts relative to OOF. LightGBM and XGBoost generalise better to the public test set than CatBoost.

### Private Leaderboard RMSE

| Model | Private LB |
|-------|------------|
| LightGBM | 0.5676 |
| XGBoost | 0.5688 |
| **CatBoost** | **0.5635** |

The ranking inverts again on the private leaderboard — CatBoost recovers to first place, matching the OOF ranking order. LGBM and XGB remain closely matched.

---

## Ensemble Performance

### OOF RMSE of blended predictions (equal weights)

| Combination | OOF RMSE |
|-------------|----------|
| LGBM alone | 0.6024 |
| XGB alone | 0.6056 |
| CatBoost alone | 0.5984 |
| LGBM + XGB | 0.6031 |
| **LGBM + CatBoost** | **0.5983** |
| XGB + CatBoost | 0.5997 |
| LGBM + XGB + CatBoost | 0.5998 |

### Public Leaderboard RMSE

| Combination | Public LB |
|-------------|-----------|
| LGBM alone | 0.5856 |
| XGB alone | 0.5859 |
| CatBoost alone | 0.5909 |
| **LGBM + XGB** | **0.5853** |
| LGBM + CatBoost | 0.5866 |
| XGB + CatBoost | 0.5865 |
| LGBM + XGB + CatBoost | 0.5857 |

### Private Leaderboard RMSE

| Combination | Private LB |
|-------------|------------|
| LGBM alone | 0.5676 |
| XGB alone | 0.5688 |
| **CatBoost alone** | **0.5635** |
| LGBM + XGB | 0.5677 |
| LGBM + CatBoost | 0.5637 |
| XGB + CatBoost | 0.5641 |
| LGBM + XGB + CatBoost | 0.5647 |

On the private leaderboard, **CatBoost alone is the best single model** and any ensemble including CatBoost outperforms LGBM+XGB.

---

## OOF Prediction Correlation

Pearson correlation between the mean OOF predictions of each model pair:

| Pair | r |
|------|---|
| LGBM vs XGB | 0.9968 |
| LGBM vs CatBoost | 0.9922 |
| XGB vs CatBoost | 0.9916 |

All three models are extremely highly correlated (>0.99). Blending provides very little diversity benefit — the gains from ensembling are marginal in all three evaluation contexts.

---

## Blend Weight Analysis

A grid search over 201 weight values for the LGBM/XGB blend (OOF-based) finds:

- **Optimal weight:** `w_lgbm = 0.96` (nearly pure LGBM)
- **OOF RMSE at optimal weight:** 0.602424 vs. 0.602430 at equal weights

The improvement is negligible — consistent with the very high inter-model correlation.

---

## Three-Way Ranking Analysis

The table below tracks each model/ensemble's rank across all three evaluation contexts:

| Combination | OOF rank | Public rank | Private rank |
|-------------|----------|-------------|--------------|
| CatBoost alone | 1 | 6 | **1** |
| LGBM + CatBoost | 2 | 5 | **2** |
| LGBM + XGB + CatBoost | 3 | 3 | **4** |
| XGB + CatBoost | 4 | 4 | **3** |
| LGBM alone | 5 | **1** | 5 |
| LGBM + XGB | 6 | **2** | 6 |
| XGB alone | 7 | 7 | 7 |

The pattern is striking: OOF and private LB agree closely on ranking; public LB is the outlier. This has a direct implication — **the public leaderboard was not a trustworthy signal for model selection in this competition**.

---

## Analysis: Why Do Rankings Diverge Across Evaluation Contexts?

### The public leaderboard was an unreliable signal

The public leaderboard was computed on approximately 30% of the test set. With ~750 essays (estimated), RMSE differences of 0.001–0.005 between models are within the noise of a small sample. The apparent public ranking — favouring LGBM over CatBoost — was likely a statistical artifact of which specific essays landed in the public split, rather than a true generalisation difference.

The private leaderboard (the remaining ~70% of test essays) reverses this ranking and aligns with OOF, suggesting that OOF was in fact a better predictor of final performance than the public leaderboard.

### The private test set likely shares distributional properties with training data

CatBoost's performance pattern tells a coherent story:
- **Best OOF** (0.5984): optimised against training distribution via cross-validation
- **Worst public LB** (0.5909): public split may have had a different score distribution or essay characteristics
- **Best private LB** (0.5635): private split is larger and more representative, matching the training distribution

This is consistent with the private test set being drawn from the same population and prompt as the training essays — the full Kaggle test set across both splits represents the same cohort. The public 30% split happened to be less representative, while the private 70% aligns with what the models were trained on.

### CatBoost's OOF optimism was real but masked by an unrepresentative public split

Our earlier conclusion — that CatBoost's OOF was overly optimistic and that it overfitted — was correct in the context of the public leaderboard. But the private results suggest the opposite: CatBoost's strong OOF was a *genuine* signal that was obscured by public split noise, not overfitting. CatBoost's symmetric trees, while less expressive, may provide stronger regularisation that happens to suit the score distribution in this dataset.

### The OOF/CV estimate was reliable overall

All three models improve by ~0.033–0.037 RMSE from OOF to private LB, a consistent gap attributable to the CV overhead (models trained on 90% of data in each fold vs. the full training set). The *relative ordering* of models — CatBoost best, XGBoost worst — is preserved between OOF and private LB. This validates the CV setup as a reliable ranking signal, despite not being reliable on the public split.

---

## Conclusions

### 1. Trust OOF over the public leaderboard for model selection

In this competition the public leaderboard (30% of test) was a noisier ranking signal than OOF (5-seed × 10-fold on the full training set). The OOF ranking and private LB ranking agree; the public LB ranking does not. For future work, use OOF as the primary selection criterion and treat public LB scores as approximate confirmation rather than ground truth.

### 2. CatBoost is the strongest single model; LGBM+XGB is a false optimum

The model selection mistake was over-optimising for the public LB. LGBM+XGB appeared best publicly (0.5853) but placed 6th privately (0.5677). CatBoost alone scored best privately (0.5635). Any ensemble including CatBoost would have been the better submission choice. The public LB led to choosing the wrong blend.

### 3. Model diversity is the limiting factor for ensembling

With inter-model correlations above 0.99, all blending gains are marginal across all three evaluation contexts. No ensemble achieves more than ~0.003 RMSE improvement over the best individual model in any context. To benefit from ensembling, future work must introduce models with genuinely different inductive biases — different feature sets, sequence models on raw keystroke streams, or pre-trained language models on the reconstructed essay text.

### 4. The feature set is effective but saturated within tree models

The OOF-to-private improvement of ~0.035 RMSE suggests models trained on the full dataset would score around 0.560–0.565 — consistent with what we observe. The gap between best private score (0.5635) and a naive mean-prediction baseline (~0.68) is ~0.117 RMSE — substantial signal extracted from the keystroke logs. However, the ceiling within the current feature set and model class appears to be around 0.56 RMSE.

### 5. Future directions

- **Reconstruct and evaluate essay text directly**: The reconstructed essay is available at inference time. A lightweight LM (e.g., DeBERTa fine-tuned on essay scoring) applied to the reconstructed text would likely push well below 0.55 RMSE. Tree models cannot access the semantic content of the text beyond what TF-IDF SVD captures.
- **Sequence modelling on keystroke streams**: Transformer or LSTM models trained directly on the raw event sequence could discover temporal patterns that aggregate features miss — particularly how writing behaviour *changes* over the session (acceleration, hesitation arcs, revision clusters).
- **Richer revision features**: Current revision features treat all deletions equally. Distinguishing local typo corrections (single-character backspaces) from structural revisions (multi-word Remove/Cut) and tracking whether deletions are immediately followed by re-insertion of similar text would add meaningful signal.
- **Use CatBoost or CatBoost-inclusive ensembles** as the primary submission, validated by OOF rather than public LB.

---

## Best Submissions

| Context | Best submission | RMSE |
|---------|----------------|------|
| Public LB | `predict_lgbm_xgb_ensemble.py` | 0.5853 |
| Private LB | `predict_catboost.py` | 0.5635 |
| OOF (proxy) | `predict_catboost.py` | 0.5984 |
