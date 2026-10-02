"""
SmartPredict-PdM
Stage 4 — Ensemble Learning Experiment
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
    MODELS_DIR
)

from preprocessing import (
    build_preprocessor,
    build_feature_frame
)


warnings.filterwarnings("ignore")


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(RESULTS_TABLE_DIR, exist_ok=True)
os.makedirs(RESULTS_FIGURE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("ENSEMBLE LEARNING EXPERIMENT")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# Build feature matrix using existing preprocessing function
X = build_feature_frame(df)


# ============================================================
# CRITICAL TARGET LEAKAGE FIX
# ============================================================

# Machine failure is the target.
# It must NEVER be used as an input feature.

if TARGET_COLUMN in X.columns:
    X = X.drop(
        columns=[TARGET_COLUMN]
    )


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

# Existing project function takes NO arguments.
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
# ENSEMBLE MODELS
# ============================================================

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),


    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),


    "AdaBoost":
        AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        ),


    "XGBoost":
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
}


# ============================================================
# VOTING ENSEMBLE
# ============================================================

voting_model = VotingClassifier(

    estimators=[

        (
            "rf",

            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),

        (
            "extra",

            ExtraTreesClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),

        (
            "xgb",

            XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ],

    voting="soft"
)


models[
    "Voting Ensemble"
] = voting_model


# ============================================================
# STACKING ENSEMBLE
# ============================================================

stacking_model = StackingClassifier(

    estimators=[

        (
            "rf",

            RandomForestClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),

        (
            "extra",

            ExtraTreesClassifier(
                n_estimators=150,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        ),

        (
            "xgb",

            XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        )
    ],

    final_estimator=LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ),

    cv=3,

    stack_method="predict_proba",

    n_jobs=-1
)


models[
    "Stacking Ensemble"
] = stacking_model


# ============================================================
# TRAINING
# ============================================================

results = []


for name, model in models.items():

    print("\n" + "=" * 70)
    print(f"Training: {name}")
    print("=" * 70)


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


    scores = cross_validate(

        pipeline,

        X,

        y,

        cv=cv,

        scoring=scoring,

        n_jobs=-1,

        return_train_score=False
    )


    # ========================================================
    # METRICS
    # ========================================================

    precision = np.mean(
        scores["test_precision"]
    )

    recall = np.mean(
        scores["test_recall"]
    )

    f1 = np.mean(
        scores["test_f1"]
    )

    roc_auc = np.mean(
        scores["test_roc_auc"]
    )

    pr_auc = np.mean(
        scores["test_pr_auc"]
    )

    mcc = np.mean(
        scores["test_mcc"]
    )


    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"MCC       : {mcc:.4f}"
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    results.append({

        "Model": name,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": roc_auc,

        "PR_AUC": pr_auc,

        "MCC": mcc
    })


    # ========================================================
    # FINAL FIT
    # ========================================================

    print(
        "\nTraining final model on complete dataset..."
    )

    pipeline.fit(
        X,
        y
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    filename = (

        name
        .lower()
        .replace(" ", "_")
        + ".joblib"
    )


    model_path = os.path.join(

        MODELS_DIR,

        filename
    )


    joblib.dump(
        pipeline,
        model_path
    )


    print(
        f"Saved: {model_path}"
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "PR_AUC",
    ascending=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")

print("=" * 70)

print("ENSEMBLE RESULTS")

print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(

    RESULTS_TABLE_DIR,

    "ensemble_model_results.csv"
)


results_df.to_csv(
    output_path,
    index=False
)


print(
    f"\nResults saved to: {output_path}"
)


# ============================================================
# GRAPH FUNCTION
# ============================================================

def save_metric_plot(
    dataframe,
    metric,
    title,
    filename,
    ylabel
):

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        dataframe["Model"],
        dataframe[metric]
    )

    plt.xlabel(
        "Ensemble Model"
    )

    plt.ylabel(
        ylabel
    )

    plt.title(
        title
    )

    plt.ylim(
        0,
        1
    )

    plt.xticks(
        rotation=25
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            RESULTS_FIGURE_DIR,
            filename
        ),

        dpi=300
    )

    plt.close()


# ============================================================
# SAVE GRAPHS
# ============================================================

save_metric_plot(

    results_df,

    "PR_AUC",

    "Ensemble Models — PR-AUC Comparison",

    "ensemble_pr_auc_comparison.png",

    "PR-AUC"
)


save_metric_plot(

    results_df,

    "F1",

    "Ensemble Models — F1 Comparison",

    "ensemble_f1_comparison.png",

    "F1 Score"
)


save_metric_plot(

    results_df,

    "Recall",

    "Ensemble Models — Failure Recall",

    "ensemble_recall_comparison.png",

    "Failure Recall"
)


save_metric_plot(

    results_df,

    "MCC",

    "Ensemble Models — MCC Comparison",

    "ensemble_mcc_comparison.png",

    "Matthews Correlation Coefficient"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("ENSEMBLE EXPERIMENT COMPLETED")

print("=" * 70)

print("\nGenerated:")

print("✓ Random Forest")
print("✓ Extra Trees")
print("✓ AdaBoost")
print("✓ XGBoost")
print("✓ Voting Ensemble")
print("✓ Stacking Ensemble")

print("✓ Cross-validation results")
print("✓ Saved trained models")
print("✓ PR-AUC graph")
print("✓ F1 graph")
print("✓ Recall graph")
print("✓ MCC graph")