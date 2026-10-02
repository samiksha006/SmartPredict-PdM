"""
SmartPredict-PdM
Stage 6 — Imbalance Handling Experiment
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR
)

from preprocessing import (
    build_preprocessor,
    build_feature_frame
)

warnings.filterwarnings("ignore")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("IMBALANCE HANDLING EXPERIMENT")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

X = build_feature_frame(df)

# Prevent target leakage
if TARGET_COLUMN in X.columns:
    X = X.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

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
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# SCORING
# ============================================================

scoring = {

    "precision": make_scorer(
        precision_score,
        zero_division=0
    ),

    "recall": make_scorer(
        recall_score,
        zero_division=0
    ),

    "f1": make_scorer(
        f1_score,
        zero_division=0
    ),

    "roc_auc": "roc_auc",

    "pr_auc": "average_precision",

    "mcc": make_scorer(
        matthews_corrcoef
    )
}


# ============================================================
# COMMON XGBOOST
# ============================================================

def create_xgboost():

    return XGBClassifier(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.05,

        subsample=0.9,

        colsample_bytree=0.9,

        eval_metric="logloss",

        random_state=42,

        n_jobs=-1
    )


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = {

    "XGBoost Baseline":

        create_xgboost(),


    "XGBoost Class Weighted":

        XGBClassifier(

            n_estimators=300,

            max_depth=4,

            learning_rate=0.05,

            subsample=0.9,

            colsample_bytree=0.9,

            scale_pos_weight=(
                (y == 0).sum() /
                (y == 1).sum()
            ),

            eval_metric="logloss",

            random_state=42,

            n_jobs=-1
        )
}


# ============================================================
# TRAIN BASELINE + CLASS WEIGHTED
# ============================================================

results = []


for name, model in experiments.items():

    print("\n" + "=" * 70)

    print(
        f"Training: {name}"
    )

    print("=" * 70)


    pipeline = SklearnPipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])


    scores = cross_validate(

        pipeline,

        X,

        y,

        cv=cv,

        scoring=scoring,

        n_jobs=-1,

        return_train_score=False
    )


    result = {

        "Model": name,

        "Precision":
            np.mean(
                scores["test_precision"]
            ),

        "Recall":
            np.mean(
                scores["test_recall"]
            ),

        "F1":
            np.mean(
                scores["test_f1"]
            ),

        "ROC_AUC":
            np.mean(
                scores["test_roc_auc"]
            ),

        "PR_AUC":
            np.mean(
                scores["test_pr_auc"]
            ),

        "MCC":
            np.mean(
                scores["test_mcc"]
            )
    }


    results.append(result)


    print(
        f"Precision : {result['Precision']:.4f}"
    )

    print(
        f"Recall    : {result['Recall']:.4f}"
    )

    print(
        f"F1        : {result['F1']:.4f}"
    )

    print(
        f"ROC-AUC   : {result['ROC_AUC']:.4f}"
    )

    print(
        f"PR-AUC    : {result['PR_AUC']:.4f}"
    )

    print(
        f"MCC       : {result['MCC']:.4f}"
    )


# ============================================================
# SMOTE EXPERIMENT
# ============================================================

print("\n" + "=" * 70)

print("Training: XGBoost + SMOTE")

print("=" * 70)


smote_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote",
        SMOTE(
            random_state=42,
            sampling_strategy="auto"
        )
    ),

    (
        "model",
        create_xgboost()
    )
])


scores = cross_validate(

    smote_pipeline,

    X,

    y,

    cv=cv,

    scoring=scoring,

    n_jobs=-1,

    return_train_score=False
)


smote_result = {

    "Model": "XGBoost + SMOTE",

    "Precision":
        np.mean(
            scores["test_precision"]
        ),

    "Recall":
        np.mean(
            scores["test_recall"]
        ),

    "F1":
        np.mean(
            scores["test_f1"]
        ),

    "ROC_AUC":
        np.mean(
            scores["test_roc_auc"]
        ),

    "PR_AUC":
        np.mean(
            scores["test_pr_auc"]
        ),

    "MCC":
        np.mean(
            scores["test_mcc"]
        )
}


results.append(
    smote_result
)


print(
    f"Precision : {smote_result['Precision']:.4f}"
)

print(
    f"Recall    : {smote_result['Recall']:.4f}"
)

print(
    f"F1        : {smote_result['F1']:.4f}"
)

print(
    f"ROC-AUC   : {smote_result['ROC_AUC']:.4f}"
)

print(
    f"PR-AUC    : {smote_result['PR_AUC']:.4f}"
)

print(
    f"MCC       : {smote_result['MCC']:.4f}"
)


# ============================================================
# SMOTE-TOMEK
# ============================================================

print("\n" + "=" * 70)

print("Training: XGBoost + SMOTE-Tomek")

print("=" * 70)


smotetomek_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote_tomek",
        SMOTETomek(
            random_state=42
        )
    ),

    (
        "model",
        create_xgboost()
    )
])


scores = cross_validate(

    smotetomek_pipeline,

    X,

    y,

    cv=cv,

    scoring=scoring,

    n_jobs=-1,

    return_train_score=False
)


smotetomek_result = {

    "Model": "XGBoost + SMOTE-Tomek",

    "Precision":
        np.mean(
            scores["test_precision"]
        ),

    "Recall":
        np.mean(
            scores["test_recall"]
        ),

    "F1":
        np.mean(
            scores["test_f1"]
        ),

    "ROC_AUC":
        np.mean(
            scores["test_roc_auc"]
        ),

    "PR_AUC":
        np.mean(
            scores["test_pr_auc"]
        ),

    "MCC":
        np.mean(
            scores["test_mcc"]
        )
}


results.append(
    smotetomek_result
)


print(
    f"Precision : {smotetomek_result['Precision']:.4f}"
)

print(
    f"Recall    : {smotetomek_result['Recall']:.4f}"
)

print(
    f"F1        : {smotetomek_result['F1']:.4f}"
)

print(
    f"ROC-AUC   : {smotetomek_result['ROC_AUC']:.4f}"
)

print(
    f"PR-AUC    : {smotetomek_result['PR_AUC']:.4f}"
)

print(
    f"MCC       : {smotetomek_result['MCC']:.4f}"
)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n")

print("=" * 70)

print("IMBALANCE EXPERIMENT RESULTS")

print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(

    RESULTS_TABLE_DIR,

    "imbalance_experiment_results.csv"
)


results_df.to_csv(

    output_path,

    index=False
)


print(
    f"\nResults saved to:"
)

print(
    output_path
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("STAGE 6 COMPLETED")

print("=" * 70)