# Results

All scores are RMSE (lower is better). The target is a holistic essay score on the scale 0.5 – 6.0.

---

## What is OOF RMSE?

### How it is computed

The CV scheme is **5 seeds × 10 folds = 50 models** per algorithm. For each seed:

1. The 2,471 training essays are split into 10 equal folds, stratified by score (so each fold has the same proportion of each score level).
2. For fold 1: train on folds 2–10, predict on fold 1.
3. For fold 2: train on folds 1, 3–10, predict on fold 2.
4. Repeat for all 10 folds.

After all 10 folds, **every essay has exactly one prediction** — made by a model that never saw it during training. This is the Out-of-Fold (OOF) prediction. Repeat across 5 seeds, then average the 5 OOF arrays into one mean prediction per essay. Compute RMSE against the true scores.

### What it means intuitively

- The score scale runs 0.5 – 6.0 (a range of 5.5 points, in 0.5-point increments).
- An OOF RMSE of **0.60** means the average prediction error is roughly **±0.6 score points** — about one full score band.
- If the true score is 3.0, the model typically predicts somewhere in the range 2.4 – 3.6.
- Because OOF predictions are never made on training data, OOF RMSE is an **unbiased estimate** of generalisation performance.

### Why OOF RMSE is lower on the private leaderboard (~0.56)

- Each OOF model is trained on only **90% of the data** (9 folds out of 10).
- The Kaggle submission averages 50 fold models — all trained on overlapping subsets of the full training set, giving more effective training data.
- The consistent ~0.035 improvement from OOF to private LB across all three models confirms this is a systematic effect of training set size, not overfitting.

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

- CatBoost achieves the best OOF score — its symmetric tree structure provides stronger regularisation, which suits this dataset.
- XGBoost is the weakest of the three, despite having Optuna-tuned hyperparameters.

### Public Leaderboard RMSE

| Model | Public LB |
|-------|-----------|
| **LightGBM** | **0.5856** |
| XGBoost | 0.5859 |
| CatBoost | 0.5909 |

- The ranking **inverts** relative to OOF — CatBoost drops to last, LightGBM rises to first.
- This inversion is suspicious and a sign that the public leaderboard is noisy (see analysis below).

### Private Leaderboard RMSE

| Model | Private LB |
|-------|------------|
| LightGBM | 0.5676 |
| XGBoost | 0.5688 |
| **CatBoost** | **0.5635** |

- The ranking **inverts again** — CatBoost returns to first place, matching the OOF order.
- LGBM and XGB remain closely matched throughout.
- The OOF ranking and private LB ranking are **identical**: CatBoost > LightGBM > XGBoost.

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

- On the private leaderboard, **CatBoost alone is the best submission** — no ensemble beats it.
- Any ensemble including CatBoost outperforms LGBM+XGB on the private LB.
- The best ensemble (LGBM+CatBoost, 0.5637) is only 0.0002 behind CatBoost alone — blending adds almost nothing.

---

## OOF Prediction Correlation

Pearson correlation between the mean OOF predictions of each model pair:

| Pair | r |
|------|---|
| LGBM vs XGB | 0.9968 |
| LGBM vs CatBoost | 0.9922 |
| XGB vs CatBoost | 0.9916 |

- All three pairs have r > 0.99 — the models are **nearly identical in their predictions**.
- When two models always agree, averaging them cannot improve much: there is no diversity to exploit.
- The best ensemble improves on the best single model by at most **~0.003 RMSE** in any evaluation context — consistent with the near-zero diversity.
- This is the fundamental ceiling on ensembling with these three models on these features.

---

## Blend Weight Analysis

A grid search over 201 weight values for the LGBM/XGB blend (OOF-based) finds:

- **Optimal weight:** `w_lgbm = 0.96` (nearly pure LGBM)
- **OOF RMSE at optimal weight:** 0.602424 vs. 0.602430 at equal weights

- The improvement is **0.000006 RMSE** — effectively zero.
- The optimal weight being 0.96 says: the best use of XGB is to contribute 4% of the final prediction.
- This confirms the correlation result: LGBM and XGB are so similar that the blend collapses toward the better single model.

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

Key observations:

- **OOF and private LB agree closely.** The top 4 in OOF are the top 4 in private LB (in the same order, except positions 3 and 4 swap). The bottom 3 are the same in both.
- **Public LB is the outlier.** The top 2 publicly (LGBM, LGBM+XGB) rank 5th and 6th privately. The top 2 privately (CatBoost, LGBM+CatBoost) rank 6th and 5th publicly.
- **The public leaderboard was not a trustworthy signal for model selection in this competition.**

---

## Analysis: Why Do Rankings Diverge Across Evaluation Contexts?

### The public leaderboard was an unreliable signal

- The public leaderboard was computed on approximately **30% of the test set** (~750 essays, estimated).
- RMSE differences of 0.001–0.005 between models on a sample of 750 essays are **within the noise** of random sampling — a different 30% split could have produced a different ranking.
- The apparent public advantage of LGBM over CatBoost was most likely a **statistical artifact** of which specific essays landed in the public split, not a genuine generalisation difference.
- The private leaderboard, using the remaining ~70% (~1,750 essays), reverses the ranking — a larger sample is more stable and more representative.

### OOF was a better predictor of final performance than the public leaderboard

- The OOF ranking (CatBoost 1st, XGB 3rd) matches the private LB ranking exactly.
- The public LB ranking (LGBM 1st, CatBoost last) disagrees with both.
- Conclusion: **trust OOF over public LB** when the public split is small relative to training data. OOF uses the full training set with proper held-out folds — it is more statistically stable than a small public test sample.

### The private test set likely shares distributional properties with training data

CatBoost's performance pattern tells a coherent story:

- **Best OOF (0.5984):** CatBoost generalises well to held-out training essays — its symmetric tree structure regularises more aggressively than LGBM or XGB.
- **Worst public LB (0.5909):** The public split (~30%) happened to contain essays or score distributions that were less favourable for CatBoost — a small-sample artifact.
- **Best private LB (0.5635):** The private split (~70%) is large enough to be representative of the true test distribution, which aligns with the training distribution.
- The consistent **~0.035 RMSE improvement** from OOF to private LB across all three models (0.6024→0.5676, 0.6056→0.5688, 0.5984→0.5635) confirms all models benefit equally from the larger effective training set at submission time — the relative ordering is preserved, only the absolute level shifts.

### CatBoost's strong OOF was a genuine signal, not overfitting

- Our initial interpretation — that CatBoost's better OOF might reflect overfitting to the training distribution — turned out to be wrong.
- The private LB confirms CatBoost's OOF advantage was **real**: it generalises better to unseen data from the same population.
- CatBoost's symmetric (oblivious) trees, while less expressive than LGBM's leaf-wise growth, provide stronger regularisation that happens to suit this dataset's score distribution.

---

## Conclusions

### 1. Trust OOF over the public leaderboard for model selection

- In this competition, OOF (5-seed × 10-fold on the full training set) was a **more reliable ranking signal** than the public leaderboard (30% of test essays).
- OOF ranking and private LB ranking agree; public LB ranking disagrees with both.
- For future work: use OOF as the primary selection criterion and treat public LB scores as approximate confirmation rather than ground truth — especially when the public split is small.

### 2. CatBoost is the strongest single model; LGBM+XGB was a false optimum

- LGBM+XGB appeared best publicly (0.5853) but placed 6th privately (0.5677).
- CatBoost alone scored best privately (0.5635).
- The mistake was over-optimising for the public LB. Any ensemble including CatBoost would have been the better submission choice.
- **Lesson:** never select your final submission based solely on a small public split.

### 3. Model diversity is the limiting factor for ensembling

- Inter-model correlations exceed 0.99 for all pairs — the three models make nearly identical predictions.
- No ensemble achieves more than **~0.003 RMSE improvement** over the best individual model in any evaluation context.
- To benefit from ensembling, future work must introduce models with **genuinely different inductive biases**: different feature sets, sequence models on raw keystroke streams, or pre-trained language models on reconstructed essay text.
- Blending three nearly-identical models is not ensembling — it is averaging noise.

### 4. The feature set is effective but saturated within tree models

- The OOF-to-private improvement of ~0.035 RMSE suggests models trained on the full dataset would score around 0.560–0.565 — consistent with what we observe (0.5635).
- The gap between best private score (0.5635) and a naive mean-prediction baseline (~0.68) is ~0.117 RMSE — substantial signal is extracted from the keystroke logs.
- However, the **ceiling within the current feature set and model class appears to be around 0.56 RMSE**. The three models converge to similar predictions, suggesting the 226 features have been effectively exhausted by gradient-boosted trees.

### 5. Future directions

- **Reconstruct and score essay text directly:** The reconstructed essay is available at inference time. A lightweight language model (e.g., DeBERTa fine-tuned on essay scoring) applied to the reconstructed text would likely push well below 0.55 RMSE. Tree models cannot access the semantic content of the text beyond what TF-IDF SVD captures.
- **Sequence modelling on keystroke streams:** Transformer or LSTM models trained directly on the raw event sequence could discover temporal patterns that aggregate features miss — particularly how writing behaviour *changes* over the session (acceleration, hesitation arcs, revision clusters).
- **Richer revision features:** Current revision features treat all deletions equally. Distinguishing local typo corrections (single-character backspaces) from structural revisions (multi-word Remove/Cut) and tracking whether deletions are immediately followed by re-insertion of similar text would add meaningful signal.
- **Introduce genuinely diverse models** before ensembling — only then will blending produce meaningful gains.

---

## Best Submissions

| Context | Best submission | RMSE |
|---------|----------------|------|
| Public LB | `predict_lgbm_xgb_ensemble.py` | 0.5853 |
| Private LB | `predict_catboost.py` | 0.5635 |
| OOF (proxy) | `predict_catboost.py` | 0.5984 |
