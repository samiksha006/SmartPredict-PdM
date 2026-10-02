import sys
from pathlib import Path

# Allow imports from src
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix
)

from config import (
    DATA_PATH,
    TARGET,
    RANDOM_STATE,
    TABLES_DIR,
    FIGURES_DIR
)

from preprocessing import build_feature_frame, build_preprocessor


# ============================================================
# Configuration
# ============================================================

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# Evaluation scorer
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
# Load data
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("KERNEL MACHINE EXPERIMENT")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

X = build_feature_frame(df)
y = df[TARGET].astype(int)

print(f"Features used: {list(X.columns)}")
print(f"Failure rate: {y.mean() * 100:.2f}%")


# ============================================================
# Build preprocessing pipeline
# ============================================================

preprocessor = build_preprocessor()


# ============================================================
# Define SVM models
# ============================================================

models = {

    "Linear SVM": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Polynomial SVM": SVC(
        kernel="poly",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "RBF SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}


# ============================================================
# Hyperparameter grids
# ============================================================

param_grids = {

    "Linear SVM": {
        "model__C": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    },

    "Polynomial SVM": {
        "model__C": [
            0.1,
            1,
            10
        ],

        "model__degree": [
            2,
            3,
            4
        ],

        "model__gamma": [
            "scale",
            0.01,
            0.1
        ]
    },

    "RBF SVM": {
        "model__C": [
            0.1,
            1,
            10,
            100
        ],

        "model__gamma": [
            "scale",
            0.001,
            0.01,
            0.1
        ]
    }
}


# ============================================================
# Run experiments
# ============================================================

all_results = []
best_models = {}


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        scoring=scoring,
        refit="pr_auc",
        cv=CV,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    search.fit(X, y)

    best_models[model_name] = search.best_estimator_

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        f"\nBest PR-AUC: "
        f"{search.best_score_:.4f}"
    )

    # Get results from CV
    cv_results = pd.DataFrame(search.cv_results_)

    best_index = search.best_index_

    result = {
        "Model": model_name,
        "Best Parameters": str(search.best_params_),
        "Precision": cv_results.loc[
            best_index, "mean_test_precision"
        ],
        "Recall": cv_results.loc[
            best_index, "mean_test_recall"
        ],
        "F1": cv_results.loc[
            best_index, "mean_test_f1"
        ],
        "ROC_AUC": cv_results.loc[
            best_index, "mean_test_roc_auc"
        ],
        "PR_AUC": cv_results.loc[
            best_index, "mean_test_pr_auc"
        ],
        "MCC": cv_results.loc[
            best_index, "mean_test_mcc"
        ]
    }

    all_results.append(result)


# ============================================================
# Results table
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    "PR_AUC",
    ascending=False
)

print("\n")
print("=" * 70)
print("KERNEL MACHINE RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "Model",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC",
            "MCC"
        ]
    ].to_string(index=False)
)


# ============================================================
# Save results
# ============================================================

output_file = TABLES_DIR / "kernel_machine_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print(
    f"\nResults saved to:\n{output_file}"
)


# ============================================================
# Save best models
# ============================================================

import joblib

for model_name, model in best_models.items():

    filename = (
        model_name
        .lower()
        .replace(" ", "_")
        + "_optimized.joblib"
    )

    joblib.dump(
        model,
        Path(__file__).resolve().parents[1]
        / "models"
        / filename
    )


# ============================================================
# Visualization — F1
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="F1"
)

plt.title(
    "Optimized SVM Kernel Comparison — F1 Score"
)

plt.xlabel("Kernel Machine")
plt.ylabel("F1 Score")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "kernel_f1_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# Visualization — Recall
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="Recall"
)

plt.title(
    "Optimized SVM Kernel Comparison — Recall"
)

plt.xlabel("Kernel Machine")
plt.ylabel("Recall")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "kernel_recall_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# Visualization — PR-AUC
# ============================================================

plt.figure(figsize=(9, 6))

sns.barplot(
    data=results_df,
    x="Model",
    y="PR_AUC"
)

plt.title(
    "Optimized SVM Kernel Comparison — PR-AUC"
)

plt.xlabel("Kernel Machine")
plt.ylabel("PR-AUC")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "kernel_pr_auc_comparison.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 70)
print("KERNEL MACHINE EXPERIMENT COMPLETED")
print("=" * 70)