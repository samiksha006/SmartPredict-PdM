from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
)

from joblib import load

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    MODELS_DIR,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
)
from preprocessing import build_feature_frame


RANDOM_STATE = 42

# Illustrative industrial cost assumptions.
# FP cost is normalized to 1.
FN_COSTS = [5, 10, 20]
FP_COST = 1

THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.01), 2)


def prepare_data():
    """Load dataset and create the same feature representation used by the project."""

    df = pd.read_csv(DATA_PATH)

    X = build_feature_frame(df)

    if TARGET_COLUMN in X.columns:
        X = X.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test


def evaluate_thresholds(y_true, probabilities):
    """Evaluate classification performance across probability thresholds."""

    rows = []

    for threshold in THRESHOLDS:

        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_true,
            predictions
        )

        cost_5 = (fn * 5) + (fp * FP_COST)
        cost_10 = (fn * 10) + (fp * FP_COST)
        cost_20 = (fn * 20) + (fp * FP_COST)

        rows.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
            "PR_AUC": average_precision_score(
                y_true,
                probabilities
            ),
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "FN_Rate": fn / (fn + tp) if (fn + tp) > 0 else 0,
            "FP_Rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "Cost_FN5": cost_5,
            "Cost_FN10": cost_10,
            "Cost_FN20": cost_20,
        })

    return pd.DataFrame(rows)


def save_threshold_plots(results):
    """Generate threshold-performance visualizations."""

    # ---------------------------------------------------------
    # Recall vs threshold
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["Threshold"],
        results["Recall"],
        label="Recall",
    )

    plt.plot(
        results["Threshold"],
        results["Precision"],
        label="Precision",
    )

    plt.plot(
        results["Threshold"],
        results["F1"],
        label="F1",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Performance Metrics Across Decision Thresholds")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURE_DIR / "threshold_performance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # False negatives and false positives
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["Threshold"],
        results["FN"],
        label="False Negatives",
    )

    plt.plot(
        results["Threshold"],
        results["FP"],
        label="False Positives",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Number of Predictions")
    plt.title("False-Negative vs False-Positive Trade-off")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURE_DIR / "threshold_error_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # MCC
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["Threshold"],
        results["MCC"],
        label="MCC",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("MCC")
    plt.title("MCC Across Decision Thresholds")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURE_DIR / "threshold_mcc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # ---------------------------------------------------------
    # Cost comparison
    # ---------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["Threshold"],
        results["Cost_FN5"],
        label="FN Cost = 5 × FP",
    )

    plt.plot(
        results["Threshold"],
        results["Cost_FN10"],
        label="FN Cost = 10 × FP",
    )

    plt.plot(
        results["Threshold"],
        results["Cost_FN20"],
        label="FN Cost = 20 × FP",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Illustrative Total Cost")
    plt.title("Cost Across Decision Thresholds")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURE_DIR / "threshold_cost_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def summarize_optimal_thresholds(results):
    """Identify threshold according to different decision objectives."""

    summaries = []

    # Best F1
    row = results.loc[results["F1"].idxmax()]

    summaries.append({
        "Criterion": "Maximum F1",
        "Threshold": row["Threshold"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1": row["F1"],
        "MCC": row["MCC"],
        "FP": row["FP"],
        "FN": row["FN"],
        "Cost_FN5": row["Cost_FN5"],
        "Cost_FN10": row["Cost_FN10"],
        "Cost_FN20": row["Cost_FN20"],
    })

    # Best MCC
    row = results.loc[results["MCC"].idxmax()]

    summaries.append({
        "Criterion": "Maximum MCC",
        "Threshold": row["Threshold"],
        "Precision": row["Precision"],
        "Recall": row["Recall"],
        "F1": row["F1"],
        "MCC": row["MCC"],
        "FP": row["FP"],
        "FN": row["FN"],
        "Cost_FN5": row["Cost_FN5"],
        "Cost_FN10": row["Cost_FN10"],
        "Cost_FN20": row["Cost_FN20"],
    })

    # Minimum cost for each scenario
    for cost_column, label in [
        ("Cost_FN5", "Minimum Cost — FN = 5×FP"),
        ("Cost_FN10", "Minimum Cost — FN = 10×FP"),
        ("Cost_FN20", "Minimum Cost — FN = 20×FP"),
    ]:

        row = results.loc[results[cost_column].idxmin()]

        summaries.append({
            "Criterion": label,
            "Threshold": row["Threshold"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "F1": row["F1"],
            "MCC": row["MCC"],
            "FP": row["FP"],
            "FN": row["FN"],
            "Cost_FN5": row["Cost_FN5"],
            "Cost_FN10": row["Cost_FN10"],
            "Cost_FN20": row["Cost_FN20"],
        })

    return pd.DataFrame(summaries)


def main():

    print("=" * 70)
    print("SMARTPREDICT-PdM")
    print("Threshold Optimization + Cost Analysis")
    print("=" * 70)

    print("\nLoading dataset...")

    X_train, X_test, y_train, y_test = prepare_data()

    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}")
    print(f"Test failures    : {y_test.sum()}")

    # ---------------------------------------------------------
    # Load optimized XGBoost
    # ---------------------------------------------------------

    model_path = MODELS_DIR / "xgboost_optimized.joblib"

    if not model_path.exists():

        raise FileNotFoundError(
            f"Optimized XGBoost model not found:\n{model_path}"
        )

    print("\nLoading optimized XGBoost...")

    model = load(model_path)

    # ---------------------------------------------------------
    # Probability prediction
    # ---------------------------------------------------------

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nProbability predictions generated.")

    # ---------------------------------------------------------
    # Threshold evaluation
    # ---------------------------------------------------------

    print("\nEvaluating thresholds...")

    results = evaluate_thresholds(
        y_test,
        probabilities
    )

    threshold_path = (
        RESULTS_TABLE_DIR /
        "threshold_analysis.csv"
    )

    results.to_csv(
        threshold_path,
        index=False
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = summarize_optimal_thresholds(results)

    summary_path = (
        RESULTS_TABLE_DIR /
        "optimal_threshold_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------

    save_threshold_plots(results)

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLD SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nFiles generated:")

    print(
        f"  {threshold_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {RESULTS_FIGURE_DIR / 'threshold_performance.png'}"
    )

    print(
        f"  {RESULTS_FIGURE_DIR / 'threshold_error_tradeoff.png'}"
    )

    print(
        f"  {RESULTS_FIGURE_DIR / 'threshold_mcc.png'}"
    )

    print(
        f"  {RESULTS_FIGURE_DIR / 'threshold_cost_analysis.png'}"
    )

    print("\nThreshold and cost analysis completed successfully.")


if __name__ == "__main__":
    main()