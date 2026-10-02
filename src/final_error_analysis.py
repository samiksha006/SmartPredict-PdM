# src/final_error_analysis.py

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# -------------------------------------------------------------------
# Project path
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(BASE_DIR))

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_DIR,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
)

from src.preprocessing import (
    build_feature_frame,
    build_preprocessor,
)

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------------
# Output directories
# -------------------------------------------------------------------

ERROR_DIR = RESULTS_DIR / "final_error_analysis"
FIGURE_DIR = ERROR_DIR / "figures"

ERROR_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------

RANDOM_STATE = 42

THRESHOLDS = {
    "Default_0.50": 0.50,
    "Cost_1_to_1_0.20": 0.20,
    "Cost_5_to_1_0.11": 0.11,
    "Cost_10_to_1_0.05": 0.05,
    "Cost_20_to_1_0.05": 0.05,
}


# -------------------------------------------------------------------
# Optimized XGBoost parameters
# -------------------------------------------------------------------

XGB_PARAMS = {
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


# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------

print_header("1. LOADING DATASET")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Failure count: {df[TARGET_COLUMN].sum()}")
print(f"Normal count: {(df[TARGET_COLUMN] == 0).sum()}")


# -------------------------------------------------------------------
# Build feature matrix
# -------------------------------------------------------------------

print_header("2. BUILDING FEATURE MATRIX")

X = build_feature_frame(df)

if TARGET_COLUMN in X.columns:
    X = X.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN].astype(int)

print("Model features:")
for col in X.columns:
    print(f"  - {col}")

print(f"\nNumber of features: {X.shape[1]}")


# -------------------------------------------------------------------
# EXACT FROZEN SPLIT
#
# Full dataset
#       |
#       +---- 8000 development
#       |        |
#       |        +---- 6400 train
#       |        |
#       |        +---- 1600 validation
#       |
#       +---- 2000 untouched final test
# -------------------------------------------------------------------

print_header("3. RECREATING FROZEN TRAIN / VALIDATION / TEST SPLIT")

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_STATE,
)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Final test samples : {len(X_test)}")

print("\nFailure distribution:")
print(f"Train      : {y_train.sum()}")
print(f"Validation : {y_val.sum()}")
print(f"Test       : {y_test.sum()}")


# -------------------------------------------------------------------
# Build optimized XGBoost pipeline
# -------------------------------------------------------------------

print_header("4. TRAINING OPTIMIZED XGBOOST FROM SCRATCH")

preprocessor = build_preprocessor()

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    **XGB_PARAMS,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model),
    ]
)

model.fit(X_train, y_train)

print("Optimized XGBoost training completed.")


# -------------------------------------------------------------------
# Test probabilities
# -------------------------------------------------------------------

print_header("5. GENERATING FINAL TEST PROBABILITIES")

test_prob = model.predict_proba(X_test)[:, 1]

print(f"Minimum probability: {test_prob.min():.6f}")
print(f"Maximum probability: {test_prob.max():.6f}")
print(f"Mean probability   : {test_prob.mean():.6f}")


# -------------------------------------------------------------------
# Prepare test records
# -------------------------------------------------------------------

test_records = df.loc[X_test.index].copy()

test_records["true_label"] = y_test.values
test_records["failure_probability"] = test_prob

# Preserve original row index
test_records["original_index"] = X_test.index

test_records = test_records.reset_index(drop=True)


# -------------------------------------------------------------------
# Overall probability metrics
# -------------------------------------------------------------------

roc_auc = roc_auc_score(y_test, test_prob)
pr_auc = average_precision_score(y_test, test_prob)

print_header("6. THRESHOLD-INDEPENDENT TEST PERFORMANCE")

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")


# -------------------------------------------------------------------
# Threshold evaluation
# -------------------------------------------------------------------

print_header("7. FINAL ERROR ANALYSIS")

summary_rows = []

threshold_records = {}

for operating_point, threshold in THRESHOLDS.items():

    prediction = (test_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )
    mcc = matthews_corrcoef(y_test, prediction)

    summary_rows.append({
        "Operating Point": operating_point,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "MCC": mcc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    })

    threshold_records[operating_point] = {
        "threshold": threshold,
        "prediction": prediction,
    }

    print(f"\n{operating_point}")
    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


summary_df = pd.DataFrame(summary_rows)

summary_path = ERROR_DIR / "threshold_error_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nSaved: {summary_path}")


# -------------------------------------------------------------------
# Create detailed error records for every threshold
# -------------------------------------------------------------------

print_header("8. CREATING DETAILED ERROR RECORDS")

all_error_records = []

for operating_point, data in threshold_records.items():

    threshold = data["threshold"]
    prediction = data["prediction"]

    records = test_records.copy()

    records["operating_point"] = operating_point
    records["threshold"] = threshold
    records["prediction"] = prediction

    def classify_error(row):

        true_label = row["true_label"]
        pred = row["prediction"]

        if true_label == 1 and pred == 1:
            return "True Positive"

        elif true_label == 0 and pred == 0:
            return "True Negative"

        elif true_label == 1 and pred == 0:
            return "False Negative"

        elif true_label == 0 and pred == 1:
            return "False Positive"

        return "Unknown"

    records["error_type"] = records.apply(
        classify_error,
        axis=1
    )

    all_error_records.append(records)


all_errors_df = pd.concat(
    all_error_records,
    ignore_index=True
)

all_errors_path = ERROR_DIR / "final_error_records.csv"

all_errors_df.to_csv(
    all_errors_path,
    index=False
)

print(f"Saved: {all_errors_path}")


# -------------------------------------------------------------------
# Save false negatives and false positives
# -------------------------------------------------------------------

print_header("9. SAVING FALSE NEGATIVES AND FALSE POSITIVES")

for operating_point in THRESHOLDS:

    subset = all_errors_df[
        all_errors_df["operating_point"] == operating_point
    ]

    fn_df = subset[
        subset["error_type"] == "False Negative"
    ].copy()

    fp_df = subset[
        subset["error_type"] == "False Positive"
    ].copy()

    safe_name = operating_point.lower()

    fn_path = ERROR_DIR / f"false_negatives_{safe_name}.csv"
    fp_path = ERROR_DIR / f"false_positives_{safe_name}.csv"

    fn_df.to_csv(fn_path, index=False)
    fp_df.to_csv(fp_path, index=False)

    print(
        f"{operating_point}: "
        f"FN={len(fn_df)}, FP={len(fp_df)}"
    )


# -------------------------------------------------------------------
# Rescued failures
#
# Failures that were FN at threshold 0.50 but become TP
# at a lower threshold.
# -------------------------------------------------------------------

print_header("10. ANALYZING RESCUED FAILURES")

default_threshold = THRESHOLDS["Default_0.50"]

default_prediction = (
    test_prob >= default_threshold
).astype(int)

default_fn_mask = (
    (y_test.values == 1) &
    (default_prediction == 0)
)

rescued_summary = []

for operating_point, threshold in THRESHOLDS.items():

    if threshold >= default_threshold:
        continue

    new_prediction = (
        test_prob >= threshold
    ).astype(int)

    rescued_mask = (
        default_fn_mask &
        (new_prediction == 1)
    )

    rescued_count = rescued_mask.sum()

    rescued_summary.append({
        "Operating Point": operating_point,
        "Threshold": threshold,
        "Failures missed at 0.50": int(default_fn_mask.sum()),
        "Failures rescued": int(rescued_count),
    })

    rescued_df = test_records.loc[
        rescued_mask
    ].copy()

    rescued_df["new_threshold"] = threshold
    rescued_df["status"] = "Rescued Failure"

    rescued_path = (
        ERROR_DIR /
        f"rescued_failures_{operating_point.lower()}.csv"
    )

    rescued_df.to_csv(
        rescued_path,
        index=False
    )

    print(
        f"{operating_point}: "
        f"rescued {rescued_count} failures"
    )


rescued_summary_df = pd.DataFrame(rescued_summary)

rescued_summary_path = (
    ERROR_DIR /
    "rescued_failures_summary.csv"
)

rescued_summary_df.to_csv(
    rescued_summary_path,
    index=False
)


# -------------------------------------------------------------------
# New false alarms
#
# Normal machines that were TN at 0.50 but become FP
# after lowering threshold.
# -------------------------------------------------------------------

print_header("11. ANALYZING NEW FALSE ALARMS")

default_tn_mask = (
    (y_test.values == 0) &
    (default_prediction == 0)
)

false_alarm_summary = []

for operating_point, threshold in THRESHOLDS.items():

    if threshold >= default_threshold:
        continue

    new_prediction = (
        test_prob >= threshold
    ).astype(int)

    new_fp_mask = (
        default_tn_mask &
        (new_prediction == 1)
    )

    count = new_fp_mask.sum()

    false_alarm_summary.append({
        "Operating Point": operating_point,
        "Threshold": threshold,
        "New False Alarms": int(count),
    })

    false_alarm_df = test_records.loc[
        new_fp_mask
    ].copy()

    false_alarm_df["new_threshold"] = threshold
    false_alarm_df["status"] = "New False Alarm"

    path = (
        ERROR_DIR /
        f"new_false_alarms_{operating_point.lower()}.csv"
    )

    false_alarm_df.to_csv(
        path,
        index=False
    )

    print(
        f"{operating_point}: "
        f"{count} new false alarms"
    )


false_alarm_summary_df = pd.DataFrame(
    false_alarm_summary
)

false_alarm_summary_df.to_csv(
    ERROR_DIR / "new_false_alarm_summary.csv",
    index=False
)


# -------------------------------------------------------------------
# Feature-level analysis
# -------------------------------------------------------------------

print_header("12. FEATURE-LEVEL ERROR ANALYSIS")

feature_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Use default 0.50 threshold for detailed feature comparison

default_records = test_records.copy()

default_records["prediction"] = default_prediction

default_records["error_type"] = np.select(
    [
        (default_records["true_label"] == 1) &
        (default_records["prediction"] == 1),

        (default_records["true_label"] == 0) &
        (default_records["prediction"] == 0),

        (default_records["true_label"] == 1) &
        (default_records["prediction"] == 0),

        (default_records["true_label"] == 0) &
        (default_records["prediction"] == 1),
    ],
    [
        "True Positive",
        "True Negative",
        "False Negative",
        "False Positive",
    ],
    default="Unknown"
)


feature_summary = (
    default_records
    .groupby("error_type")[feature_columns]
    .mean()
)

feature_summary_path = (
    ERROR_DIR /
    "default_threshold_feature_means.csv"
)

feature_summary.to_csv(feature_summary_path)

print("\nMean feature values by error type:")
print(feature_summary)

print(
    f"\nSaved: {feature_summary_path}"
)


# -------------------------------------------------------------------
# Failure-focused analysis
# -------------------------------------------------------------------

print_header("13. FAILURE-FOCUSED ANALYSIS")

# Create predictions for ALL test samples first
default_prediction_all = (
    test_prob >= 0.50
).astype(int)

# Add prediction to the complete test dataframe
failure_analysis_df = test_records.copy()

failure_analysis_df["default_prediction"] = default_prediction_all

failure_analysis_df["default_status"] = np.where(
    failure_analysis_df["default_prediction"] == 1,
    "Detected",
    "Missed"
)

# Keep only actual failures
actual_failures = failure_analysis_df[
    failure_analysis_df["true_label"] == 1
].copy()

print(
    f"Actual failures in final test set: "
    f"{len(actual_failures)}"
)

print(
    f"Detected at threshold 0.50: "
    f"{(actual_failures['default_prediction'] == 1).sum()}"
)

print(
    f"Missed at threshold 0.50: "
    f"{(actual_failures['default_prediction'] == 0).sum()}"
)


# -------------------------------------------------------------------
# Compare detected vs missed failures
# -------------------------------------------------------------------

failure_summary = (
    actual_failures
    .groupby("default_status")[feature_columns]
    .mean()
)

failure_summary_path = (
    ERROR_DIR /
    "failure_detected_vs_missed_feature_means.csv"
)

failure_summary.to_csv(
    failure_summary_path
)

print("\nMean feature values for actual failures:")
print(failure_summary)

print(
    f"\nSaved: {failure_summary_path}"
)


# -------------------------------------------------------------------
# Detailed missed-failure records
# -------------------------------------------------------------------

missed_failures = actual_failures[
    actual_failures["default_prediction"] == 0
].copy()

missed_failures = missed_failures.sort_values(
    "failure_probability",
    ascending=False
)

missed_failures_path = (
    ERROR_DIR /
    "missed_failures_default_0.50.csv"
)

missed_failures.to_csv(
    missed_failures_path,
    index=False
)

print(
    f"\nMissed failures saved: "
    f"{missed_failures_path}"
)


# -------------------------------------------------------------------
# Detected-failure records
# -------------------------------------------------------------------

detected_failures = actual_failures[
    actual_failures["default_prediction"] == 1
].copy()

detected_failures = detected_failures.sort_values(
    "failure_probability",
    ascending=False
)

detected_failures_path = (
    ERROR_DIR /
    "detected_failures_default_0.50.csv"
)

detected_failures.to_csv(
    detected_failures_path,
    index=False
)

print(
    f"Detected failures saved: "
    f"{detected_failures_path}"
)


# -------------------------------------------------------------------
# Compare probability of detected vs missed failures
# -------------------------------------------------------------------

probability_summary = (
    actual_failures
    .groupby("default_status")["failure_probability"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Minimum="min",
        Maximum="max"
    )
)

probability_summary_path = (
    ERROR_DIR /
    "detected_vs_missed_probability_summary.csv"
)

probability_summary.to_csv(
    probability_summary_path
)

print("\nFailure probability summary:")
print(probability_summary)

print(
    f"\nSaved: {probability_summary_path}"
)



# -------------------------------------------------------------------
# Probability distribution
# -------------------------------------------------------------------

print_header("14. GENERATING PROBABILITY DISTRIBUTION PLOT")

plt.figure(figsize=(10, 6))

plt.hist(
    test_prob[y_test.values == 0],
    bins=40,
    alpha=0.7,
    label="Normal machines"
)

plt.hist(
    test_prob[y_test.values == 1],
    bins=40,
    alpha=0.7,
    label="Failure machines"
)

for threshold in sorted(set(THRESHOLDS.values())):

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Threshold = {threshold:.2f}"
    )

plt.xlabel("Predicted Failure Probability")
plt.ylabel("Number of Machines")
plt.title("Failure Probability Distribution on Untouched Test Set")
plt.legend()
plt.tight_layout()

probability_plot = (
    FIGURE_DIR /
    "test_probability_distribution.png"
)

plt.savefig(
    probability_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {probability_plot}")


# -------------------------------------------------------------------
# Confusion matrix plots
# -------------------------------------------------------------------

print_header("15. GENERATING CONFUSION MATRIX PLOTS")

for operating_point, threshold in THRESHOLDS.items():

    prediction = (
        test_prob >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        prediction
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Confusion Matrix\n"
        f"{operating_point} (Threshold={threshold:.2f})"
    )

    plt.tight_layout()

    safe_name = operating_point.lower()

    path = (
        FIGURE_DIR /
        f"confusion_matrix_{safe_name}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# -------------------------------------------------------------------
# Error count comparison
# -------------------------------------------------------------------

print_header("16. ERROR COUNT COMPARISON")

plot_df = summary_df[
    ["Operating Point", "FP", "FN"]
].copy()

plot_df = plot_df.set_index(
    "Operating Point"
)

plot_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.xlabel("Operating Point")
plt.ylabel("Number of Errors")
plt.title("False Positives vs False Negatives")
plt.xticks(rotation=20)
plt.tight_layout()

error_count_plot = (
    FIGURE_DIR /
    "false_positive_vs_false_negative.png"
)

plt.savefig(
    error_count_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {error_count_plot}")


# -------------------------------------------------------------------
# Recall vs false alarms
# -------------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    summary_df["Threshold"],
    summary_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    summary_df["Threshold"],
    summary_df["Precision"],
    marker="o",
    label="Precision"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision–Recall Trade-off Across Decision Thresholds")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

tradeoff_plot = (
    FIGURE_DIR /
    "precision_recall_threshold_tradeoff.png"
)

plt.savefig(
    tradeoff_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {tradeoff_plot}")


# -------------------------------------------------------------------
# Save model probabilities
# -------------------------------------------------------------------

probability_df = test_records[
    [
        "original_index",
        "true_label",
        "failure_probability",
    ]
].copy()

probability_df = probability_df.sort_values(
    "failure_probability",
    ascending=False
)

probability_path = (
    ERROR_DIR /
    "final_test_probabilities.csv"
)

probability_df.to_csv(
    probability_path,
    index=False
)

print(f"Saved: {probability_path}")


# -------------------------------------------------------------------
# Final report
# -------------------------------------------------------------------

print_header("17. FINAL SUMMARY")

print(summary_df.to_string(index=False))

print("\n" + "-" * 80)

print(
    f"Threshold-independent ROC-AUC : {roc_auc:.4f}"
)

print(
    f"Threshold-independent PR-AUC  : {pr_auc:.4f}"
)

print(
    f"\nFailures in final test set: "
    f"{int(y_test.sum())}"
)

print(
    f"Normal machines in final test set: "
    f"{int((y_test == 0).sum())}"
)

print("\nOutput directory:")
print(ERROR_DIR)

print("\nStage 17 completed successfully.")