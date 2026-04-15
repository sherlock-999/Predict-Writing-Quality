"""
Optuna Hyperparameter Tuning for LightGBM
==========================================
How it works:
  - Optuna runs N trials. Each trial picks a different set of hyperparameters
    (using TPE — Tree-structured Parzen Estimator, a Bayesian method that
    learns which regions of the search space are promising from past trials).
  - Each trial evaluates the hyperparameters via 5-fold stratified CV and
    returns the OOF RMSE. Optuna minimises this value.
  - Pruning: trials that are clearly worse than the median are stopped early
    (MedianPruner), saving time.
  - At the end, best parameters are printed and saved to models/lgbm_best_params.json.

Usage:
    conda run -n exp python tune_lgbm.py

Output:
    models/lgbm_best_params.json   — best hyperparameters found
    lgbm_plots/optuna_history.png  — optimisation history plot
    lgbm_plots/optuna_importance.png — hyperparameter importance plot
"""

import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from writing_process.v3_features import compute_features, FEATURE_COLS

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR   = os.path.join(BASE_DIR, 'lgbm_plots_test_new')
FEATURES_FILENAME = 'train_features.csv'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Tuning config ─────────────────────────────────────────────────────────────
N_TRIALS  = 100   # number of Optuna trials
N_FOLDS   = 10     # CV folds per trial
SEED      = 42

# =============================================================================
# SECTION 1 – LOAD DATA
# =============================================================================
features_cache = os.path.join(DATA_DIR, FEATURES_FILENAME)

if os.path.exists(features_cache):
    print(f"Loading precomputed features from {features_cache}...")
    df = pd.read_csv(features_cache).fillna(0)
else:
    print("Cache not found — computing features from raw logs (run precompute_features.py to speed this up)...")
    logs   = pd.read_csv(os.path.join(DATA_DIR, 'train_logs.csv'))
    scores = pd.read_csv(os.path.join(DATA_DIR, 'train_scores.csv'))
    df = compute_features(logs).merge(scores, on='id').fillna(0)

print(f"  {df.shape[0]} essays × {len(FEATURE_COLS)} features")

X = df[FEATURE_COLS].values
y = df['score'].values

score_bins = pd.cut(y, bins=5, labels=False)
skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# =============================================================================
# SECTION 2 – OPTUNA OBJECTIVE
# =============================================================================

def objective(trial: optuna.Trial) -> float:
    """
    Each call to objective() is one Optuna trial.
    trial.suggest_* picks a value from the specified range.
    Returns OOF RMSE — Optuna minimises this.
    """
    params = {
        'objective':      'regression',
        'metric':         'rmse',
        'verbosity':      0,
        'random_state':   SEED,
        'force_col_wise': True,

        # ── Parameters being tuned ────────────────────────────────────────────
        # num_leaves: controls model complexity.
        'num_leaves':        trial.suggest_int('num_leaves', 10, 100),

        # min_child_samples: min data points required in a leaf.
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        # learning_rate: step size for gradient updates.
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.1, log=True),

        # colsample_bytree: fraction of features used per tree (XGBoost alias).
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),

        # subsample: row-level subsampling (XGBoost alias).
        'subsample':         trial.suggest_float('subsample', 0.4, 1.0),

        # reg_alpha / reg_lambda: L1 and L2 regularisation (XGBoost aliases).
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),

        # max_depth: tuned in [10, 50] — avoids underfitting from small values.
        'max_depth':         trial.suggest_int('max_depth', 10, 50),

        # min_gain_to_split: fixed at 0 — tuning this causes "no positive gain" underfitting.
        'min_gain_to_split': 0.0,

        # n_estimators: upper bound; early stopping finds the true optimum.
        'n_estimators':      2000,
    }

    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, score_bins)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_preds[val_idx] = model.predict(X_val)

        # Pruning: report intermediate value after each fold so Optuna can
        # kill unpromising trials early (before all 5 folds complete).
        fold_rmse = mean_squared_error(y[val_idx], oof_preds[val_idx]) ** 0.5
        trial.report(fold_rmse, step=fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    oof_rmse = mean_squared_error(y, oof_preds) ** 0.5
    return oof_rmse


# =============================================================================
# SECTION 3 – RUN OPTIMISATION
# =============================================================================
print(f"\nStarting Optuna study — {N_TRIALS} trials × {N_FOLDS} folds each")
print("(pruned trials are stopped early if they look worse than the median)\n")

sampler = optuna.samplers.TPESampler(seed=SEED)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2)
study   = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)

# Warm-start: add the current hand-tuned defaults as trial 0 so Optuna
# knows it needs to beat them, not wander randomly at the start.
study.enqueue_trial({
    'num_leaves':        22,
    'min_child_samples': 18,
    'learning_rate':     0.038697981947473245,
    'colsample_bytree':  0.627061253588415,
    'subsample':         0.854942238828458,
    'reg_alpha':         0.007678095440286993,
    'reg_lambda':        0.34230534302168353,
    'max_depth':         37,
})

def print_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        print(f"  Trial {trial.number:3d} | RMSE: {trial.value:.4f} "
              f"| Best: {study.best_value:.4f}")

study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_callback])

# =============================================================================
# SECTION 4 – RESULTS
# =============================================================================
print("\n" + "="*60)
print("BEST TRIAL")
print("="*60)
print(f"  OOF RMSE : {study.best_value:.4f}")
print(f"  Params   :")
for k, v in study.best_params.items():
    print(f"    {k:<25} {v}")

# Save best params to JSON so train_lgbm.py can load them
best_params_path = os.path.join(MODELS_DIR, 'lgbm_best_params.json')
with open(best_params_path, 'w') as f:
    json.dump(study.best_params, f, indent=2)
print(f"\nSaved best params to: {best_params_path}")

# =============================================================================
# SECTION 5 – PLOTS
# =============================================================================

# Plot 1 – Optimisation history (RMSE per trial)
fig, ax = plt.subplots(figsize=(10, 4))
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
trial_nums = [t.number for t in completed]
trial_vals = [t.value  for t in completed]
best_so_far = np.minimum.accumulate(trial_vals)

ax.scatter(trial_nums, trial_vals, s=15, alpha=0.5, color='steelblue', label='Trial RMSE')
ax.plot(trial_nums, best_so_far, color='red', linewidth=1.5, label='Best so far')
ax.set_xlabel('Trial')
ax.set_ylabel('OOF RMSE')
ax.set_title('Optuna Optimisation History — LightGBM', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'optuna_history.png'), dpi=150)
plt.close()
print("[Saved] optuna_history.png")

# Plot 2 – Hyperparameter importance
# fANOVA (default) can fail with AssertionError when many trials are pruned.
# Fall back to MeanDecreaseImpurity which is always stable.
try:
    importances = optuna.importance.get_param_importances(study)
except (AssertionError, RuntimeError):
    importances = optuna.importance.get_param_importances(
        study,
        evaluator=optuna.importance.MeanDecreaseImpurityImportanceEvaluator(),
    )
fig, ax = plt.subplots(figsize=(8, 5))
keys = list(importances.keys())
vals = list(importances.values())
ax.barh(keys[::-1], vals[::-1], color='steelblue', edgecolor='white')
ax.set_xlabel('Importance')
ax.set_title('Hyperparameter Importance — LightGBM', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'optuna_importance.png'), dpi=150)
plt.close()
print("[Saved] optuna_importance.png")

print("\nDone. Use lgbm_best_params.json in train_lgbm.py to retrain with best params.")
