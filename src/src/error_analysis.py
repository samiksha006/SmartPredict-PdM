
"""
SmartPredict-PdM
Stage 8: Error Analysis

Purpose:
- Analyze false positives and false negatives
- Compare XGBoost Baseline, Optimized XGBoost,
  and XGBoost + SMOTE-Tomek
- Identify feature patterns associated with errors
- Save error-analysis tables and figures

Run:
    python src\error_analysis.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
)

from preprocessing import (
    build_feature_frame,
    build_preprocessor,
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


# Optimized XGBoost parameters obtained from Stage 5
OPTIMIZED_PARAMS = {
    "n_estimators": 350,
    "max_depth": 7,
    "learning_rate": 0.022323520132149687,
    "subsample": 0.9474682096161544,
    "colsample_bytree": 0.9602378472844514,
    "min_child_weight": 1,
    "gamma": 0.14439670142040958,
    "reg_alpha": 0.0006530679464151186,
    "reg_lambda": 0.11243724759917004,
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def create_xgb(params=None):
    """
    Create XGBoost classifier.
    """

    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
        }

    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def get_error_dataframe(
    X_original,
    y_true,
    y_pred,
    y_probability,
    model_name,
):
    """
    Create a dataframe containing predictions and error types.
    """

    result = X_original.copy().reset_index(drop=True)

    result["Actual"] = np.asarray(y_true)
    result["Predicted"] = np.asarray(y_pred)
    result["Failure_Probability"] = np.asarray(y_probability)

    result["Model"] = model_name

    result["Error_Type"] = "Correct"

    # False Negative
    result.loc[
        (result["Actual"] == 1) & (result["Predicted"] == 0),
        "Error_Type"
    ] = "False Negative"

    # False Positive
    result.loc[
        (result["Actual"] == 0) & (result["Predicted"] == 1),
        "Error_Type"
    ] = "False Positive"

    # True Positive
    result.loc[
        (result["Actual"] == 1) & (result["Predicted"] == 1),
        "Error_Type"
    ] = "True Positive"

    # True Negative
    result.loc[
        (result["Actual"] == 0) & (result["Predicted"] == 0),
        "Error_Type"
    ] = "True Negative"

    return result


def print_feature_statistics(error_df, feature_columns, error_type):
    """
    Print average feature values for a specific error type.
    """

    subset = error_df[
        error_df["Error_Type"] == error_type
    ]

    print(f"\n{error_type} count: {len(subset)}")

    if len(subset) == 0:
        print("No examples found.")
        return

    print(f"\nAverage feature values for {error_type}:")

    numeric_features = [
        col
        for col in feature_columns
        if col in subset.columns
        and pd.api.types.is_numeric_dtype(subset[col])
    ]

    if numeric_features:
        stats = subset[numeric_features].mean()
        print(stats.to_string())


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("STAGE 8: ERROR ANALYSIS")
print("=" * 70)


df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# BUILD FEATURES
# ============================================================

X = build_feature_frame(df)

# Safety check:
# build_feature_frame() may retain target column.
if TARGET_COLUMN in X.columns:
    X = X.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]


print(f"\nFeatures used:")
print(list(X.columns))

print(f"\nFailure rate: {y.mean() * 100:.2f}%")


# ============================================================
# FIXED TRAIN / TEST SPLIT
# Same split strategy as Stage 7
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)


print("\n" + "=" * 70)
print("FIXED TEST SET")
print("=" * 70)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Test failures    : {int(y_test.sum())}")


# ============================================================
# ORIGINAL TEST FEATURES
# Used for human-readable error analysis
# ============================================================

X_test_original = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

preprocessor_baseline = build_preprocessor()

baseline_xgb = Pipeline(
    steps=[
        ("preprocessor", preprocessor_baseline),
        ("model", create_xgb()),
    ]
)


preprocessor_optimized = build_preprocessor()

optimized_xgb = Pipeline(
    steps=[
        ("preprocessor", preprocessor_optimized),
        ("model", create_xgb(OPTIMIZED_PARAMS)),
    ]
)


# SMOTE-Tomek pipeline
preprocessor_smote = build_preprocessor()

smote_tomek_xgb = ImbPipeline(
    steps=[
        ("preprocessor", preprocessor_smote),
        (
            "sampler",
            SMOTETomek(
                random_state=RANDOM_STATE
            ),
        ),
        ("model", create_xgb()),
    ]
)


models = {
    "XGBoost Baseline": baseline_xgb,
    "Optimized XGBoost": optimized_xgb,
    "XGBoost + SMOTE-Tomek": smote_tomek_xgb,
}


# ============================================================
# FEATURE COLUMNS FOR ANALYSIS
# ============================================================

feature_columns = list(X.columns)


# ============================================================
# STORAGE
# ============================================================

all_error_records = []
summary_records = []

false_negative_tables = {}
false_positive_tables = {}


# ============================================================
# TRAIN + ERROR ANALYSIS
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"ANALYZING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("Training model...")

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # --------------------------------------------------------
    # Error dataframe
    # --------------------------------------------------------

    error_df = get_error_dataframe(
        X_test_original,
        y_test_reset,
        y_pred,
        y_probability,
        model_name,
    )

    all_error_records.append(error_df)

    # --------------------------------------------------------
    # Extract errors
    # --------------------------------------------------------

    false_negatives = error_df[
        error_df["Error_Type"] == "False Negative"
    ].copy()

    false_positives = error_df[
        error_df["Error_Type"] == "False Positive"
    ].copy()

    false_negative_tables[model_name] = false_negatives
    false_positive_tables[model_name] = false_positives

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_records.append({
        "Model": model_name,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "False_Negative_Rate": fn / (fn + tp),
        "False_Positive_Rate": fp / (fp + tn),
        "Total_Errors": int(fp + fn),
    })

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nConfusion Matrix:")
    print(f"TN = {tn}")
    print(f"FP = {fp}")
    print(f"FN = {fn}")
    print(f"TP = {tp}")

    print(f"\nFalse Negatives: {len(false_negatives)}")
    print(f"False Positives: {len(false_positives)}")

    print_feature_statistics(
        error_df,
        feature_columns,
        "False Negative",
    )

    print_feature_statistics(
        error_df,
        feature_columns,
        "False Positive",
    )

    # --------------------------------------------------------
    # Save individual error files
    # --------------------------------------------------------

    safe_name = (
        model_name
        .lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("-", "_")
    )

    fn_path = (
        RESULTS_TABLE_DIR /
        f"{safe_name}_false_negatives.csv"
    )

    fp_path = (
        RESULTS_TABLE_DIR /
        f"{safe_name}_false_positives.csv"
    )

    false_negatives.to_csv(
        fn_path,
        index=False,
    )

    false_positives.to_csv(
        fp_path,
        index=False,
    )


# ============================================================
# COMBINED ERROR SUMMARY
# ============================================================

summary_df = pd.DataFrame(summary_records)

summary_path = (
    RESULTS_TABLE_DIR /
    "error_analysis_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# COMBINED FALSE NEGATIVES
# ============================================================

combined_fn = pd.concat(
    false_negative_tables.values(),
    ignore_index=True,
)

combined_fn_path = (
    RESULTS_TABLE_DIR /
    "false_negatives.csv"
)

combined_fn.to_csv(
    combined_fn_path,
    index=False,
)


# ============================================================
# COMBINED FALSE POSITIVES
# ============================================================

combined_fp = pd.concat(
    false_positive_tables.values(),
    ignore_index=True,
)

combined_fp_path = (
    RESULTS_TABLE_DIR /
    "false_positives.csv"
)

combined_fp.to_csv(
    combined_fp_path,
    index=False,
)


# ============================================================
# FIGURE 1: ERROR DISTRIBUTION
# ============================================================

plot_df = summary_df.set_index("Model")[
    ["FP", "FN"]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(10, 6),
)

ax.set_title(
    "False Positive and False Negative Distribution"
)

ax.set_xlabel("Model")
ax.set_ylabel("Number of Errors")

plt.xticks(
    rotation=20,
    ha="right",
)

plt.tight_layout()

error_distribution_path = (
    RESULTS_FIGURE_DIR /
    "error_distribution.png"
)

plt.savefig(
    error_distribution_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FIGURE 2: FALSE NEGATIVE FEATURE ANALYSIS
# ============================================================

numeric_features = [
    col
    for col in feature_columns
    if col in X_test_original.columns
    and pd.api.types.is_numeric_dtype(
        X_test_original[col]
    )
]


if numeric_features and len(combined_fn) > 0:

    fn_plot = combined_fn.groupby(
        "Model"
    )[numeric_features].mean()

    ax = fn_plot.plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.set_title(
        "Average Feature Values of False Negatives"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Feature Value")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    fn_feature_path = (
        RESULTS_FIGURE_DIR /
        "false_negative_features.png"
    )

    plt.savefig(
        fn_feature_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FIGURE 3: FALSE POSITIVE FEATURE ANALYSIS
# ============================================================

if numeric_features and len(combined_fp) > 0:

    fp_plot = combined_fp.groupby(
        "Model"
    )[numeric_features].mean()

    ax = fp_plot.plot(
        kind="bar",
        figsize=(12, 7),
    )

    ax.set_title(
        "Average Feature Values of False Positives"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Average Feature Value")

    plt.xticks(
        rotation=20,
        ha="right",
    )

    plt.tight_layout()

    fp_feature_path = (
        RESULTS_FIGURE_DIR /
        "false_positive_features.png"
    )

    plt.savefig(
        fp_feature_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


print("\n" + "=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(f"\nSummary:")
print(summary_path)

print("\nCombined false negatives:")
print(combined_fn_path)

print("\nCombined false positives:")
print(combined_fp_path)

print("\nFigures:")
print(
    RESULTS_FIGURE_DIR /
    "error_distribution.png"
)

print(
    RESULTS_FIGURE_DIR /
    "false_negative_features.png"
)

print(
    RESULTS_FIGURE_DIR /
    "false_positive_features.png"
)


print("\n" + "=" * 70)
print("STAGE 8 COMPLETED")
print("=" * 70)
