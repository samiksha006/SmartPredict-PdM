"""
SmartPredict-PdM
Stage 5 — XGBoost Hyperparameter Optimization using Optuna
"""

import os
import warnings
import joblib
import optuna
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR,
    MODELS_DIR
)

from preprocessing import (
    build_preprocessor,
    build_feature_frame
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_TRIALS = 30


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    RESULTS_TABLE_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("XGBOOST HYPERPARAMETER OPTIMIZATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print(
    f"\nDataset shape: {df.shape}"
)


# ============================================================
# BUILD FEATURES
# ============================================================

X = build_feature_frame(
    df
)


# ------------------------------------------------------------
# Remove target if preprocessing function returns it
# ------------------------------------------------------------

if TARGET_COLUMN in X.columns:

    X = X.drop(
        columns=[TARGET_COLUMN]
    )


y = df[
    TARGET_COLUMN
]


print(
    f"Features used: {list(X.columns)}"
)

print(
    f"Failure rate: {y.mean() * 100:.2f}%"
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = build_preprocessor()


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(

    n_splits=3,

    shuffle=True,

    random_state=RANDOM_STATE
)


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # Hyperparameter search space
    # --------------------------------------------------------

    params = {

        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500,
            step=50
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            2,
            8
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.20,
            log=True
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.00
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            1.0
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            1.0,
            log=True
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.1,
            10.0,
            log=True
        )
    }


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = XGBClassifier(

        **params,

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])


    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    scores = cross_val_score(

        pipeline,

        X,

        y,

        cv=cv,

        scoring="average_precision",

        n_jobs=-1
    )


    mean_score = np.mean(
        scores
    )


    # --------------------------------------------------------
    # Print trial result
    # --------------------------------------------------------

    print(
        f"Trial {trial.number:02d} | "
        f"PR-AUC = {mean_score:.4f}"
    )


    return mean_score


# ============================================================
# OPTUNA STUDY
# ============================================================

print("\nStarting Optuna optimization...")

print(
    f"Number of trials: {N_TRIALS}"
)

print(
    "Optimization metric: PR-AUC"
)


study = optuna.create_study(

    direction="maximize",

    study_name="SmartPredict_XGBoost_PR_AUC"
)


study.optimize(

    objective,

    n_trials=N_TRIALS,

    show_progress_bar=True
)


# ============================================================
# BEST RESULT
# ============================================================

print("\n")

print("=" * 70)

print("OPTIMIZATION COMPLETED")

print("=" * 70)


print(
    f"\nBest PR-AUC: "
    f"{study.best_value:.4f}"
)


print(
    "\nBest Parameters:"
)


for key, value in study.best_params.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# BUILD OPTIMIZED MODEL
# ============================================================

best_model = XGBClassifier(

    **study.best_params,

    eval_metric="logloss",

    random_state=RANDOM_STATE,

    n_jobs=-1
)


best_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        best_model
    )
])


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print(
    "\nTraining optimized XGBoost "
    "on complete dataset..."
)


best_pipeline.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(

    MODELS_DIR,

    "xgboost_optimized.joblib"
)


joblib.dump(

    best_pipeline,

    model_path
)


print(
    f"\nSaved optimized model:"
)

print(
    model_path
)


# ============================================================
# SAVE OPTIMIZATION RESULTS
# ============================================================

results_path = os.path.join(

    RESULTS_TABLE_DIR,

    "optimized_xgboost_results.csv"
)


optimization_results = pd.DataFrame([

    {

        "Model":
        "Optimized XGBoost",

        "PR_AUC":
        study.best_value,

        "Baseline_PR_AUC":
        0.805699,

        "Improvement":
        study.best_value - 0.805699,

        "Best_Parameters":
        str(study.best_params)
    }

])


optimization_results.to_csv(

    results_path,

    index=False
)


print(
    f"\nResults saved:"
)

print(
    results_path
)


# ============================================================
# SAVE COMPLETE OPTUNA HISTORY
# ============================================================

history_path = os.path.join(

    RESULTS_TABLE_DIR,

    "optuna_trial_history.csv"
)


trial_rows = []


for trial in study.trials:

    trial_rows.append({

        "Trial":
        trial.number,

        "PR_AUC":
        trial.value,

        "State":
        trial.state.name,

        **trial.params
    })


history_df = pd.DataFrame(
    trial_rows
)


history_df.to_csv(

    history_path,

    index=False
)


print(
    f"Trial history saved:"
)

print(
    history_path
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)

print("STAGE 5 COMPLETED")

print("=" * 70)

print(
    f"\nBaseline XGBoost PR-AUC : 0.8057"
)

print(
    f"Optimized XGBoost PR-AUC: "
    f"{study.best_value:.4f}"
)

print(
    f"Improvement             : "
    f"{study.best_value - 0.805699:+.4f}"
)

print("\nGenerated:")

print(
    "✓ Optuna optimization"
)

print(
    "✓ Optimized XGBoost model"
)

print(
    "✓ Optimization results"
)

print(
    "✓ Trial history"
)