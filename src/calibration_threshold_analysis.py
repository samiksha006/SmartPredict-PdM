from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    MODELS_DIR,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
)

from preprocessing import build_feature_frame


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

THRESHOLDS = np.round(
    np.arange(0.05, 0.96, 0.01),
    2
)

# Illustrative operational costs.
# FP cost is normalized to 1.
FN_COST_SCENARIOS = {
    "FN_5x_FP": 5,
    "FN_10x_FP": 10,
    "FN_20x_FP": 20,
}

FP_COST = 1


# ============================================================
# DATA PREPARATION
# ============================================================

def load_and_split_data():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = build_feature_frame(df)

    if TARGET_COLUMN in X.columns:
        X = X.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN]

    # --------------------------------------------------------
    # First split:
    # Development = 80%
    # Final test = 20%
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Second split:
    # Training = 80% of development
    # Validation = 20% of development
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=VALIDATION_SIZE,
        stratify=y_dev,
        random_state=RANDOM_STATE,
    )

    print(f"Total samples      : {len(df)}")
    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Final test samples : {len(X_test)}")

    print(f"\nTraining failures   : {y_train.sum()}")
    print(f"Validation failures : {y_val.sum()}")
    print(f"Test failures       : {y_test.sum()}")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# LOAD OPTIMIZED MODEL
# ============================================================

def load_optimized_model():

    model_path = MODELS_DIR / "xgboost_optimized.joblib"

    if not model_path.exists():

        raise FileNotFoundError(
            f"Optimized XGBoost model not found:\n{model_path}"
        )

    print("\nLoading optimized XGBoost model...")

    model = load(model_path)

    return model


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    mcc = matthews_corrcoef(
        y_true,
        predictions,
    )

    return {
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

def optimize_thresholds(
    y_val,
    validation_probabilities,
):

    rows = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_val,
            validation_probabilities,
            threshold,
        )

        fp = metrics["FP"]
        fn = metrics["FN"]

        metrics["Cost_FN5"] = (
            fn * 5
            + fp * FP_COST
        )

        metrics["Cost_FN10"] = (
            fn * 10
            + fp * FP_COST
        )

        metrics["Cost_FN20"] = (
            fn * 20
            + fp * FP_COST
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# SELECT THRESHOLDS
# ============================================================

def select_thresholds(results):

    selections = []

    # --------------------------------------------------------
    # Maximum F1
    # --------------------------------------------------------

    row = results.loc[
        results["F1"].idxmax()
    ]

    selections.append({
        "Criterion": "Maximum F1",
        **row.to_dict(),
    })

    # --------------------------------------------------------
    # Maximum MCC
    # --------------------------------------------------------

    row = results.loc[
        results["MCC"].idxmax()
    ]

    selections.append({
        "Criterion": "Maximum MCC",
        **row.to_dict(),
    })

    # --------------------------------------------------------
    # Cost scenarios
    # --------------------------------------------------------

    for column, label in [
        (
            "Cost_FN5",
            "Minimum Cost — FN = 5×FP",
        ),
        (
            "Cost_FN10",
            "Minimum Cost — FN = 10×FP",
        ),
        (
            "Cost_FN20",
            "Minimum Cost — FN = 20×FP",
        ),
    ]:

        row = results.loc[
            results[column].idxmin()
        ]

        selections.append({
            "Criterion": label,
            **row.to_dict(),
        })

    return pd.DataFrame(selections)


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_models(
    base_model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    print("\nTraining calibration models...")

    # --------------------------------------------------------
    # Sigmoid / Platt scaling
    # --------------------------------------------------------

    sigmoid_model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=5,
    )

    sigmoid_model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Isotonic calibration
    # --------------------------------------------------------

    isotonic_model = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=5,
    )

    isotonic_model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    sigmoid_prob = sigmoid_model.predict_proba(
        X_val
    )[:, 1]

    isotonic_prob = isotonic_model.predict_proba(
        X_val
    )[:, 1]

    # Base model probabilities
    base_prob = base_model.predict_proba(
        X_val
    )[:, 1]

    return {
        "Uncalibrated": base_prob,
        "Platt_Scaling": sigmoid_prob,
        "Isotonic": isotonic_prob,
    }


# ============================================================
# CALIBRATION METRICS
# ============================================================

def calibration_metrics(
    y_true,
    probabilities,
    model_name,
):

    return {
        "Model": model_name,

        "Brier_Score": brier_score_loss(
            y_true,
            probabilities,
        ),

        "Log_Loss": log_loss(
            y_true,
            probabilities,
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities,
        ),

        "PR_AUC": average_precision_score(
            y_true,
            probabilities,
        ),
    }


# ============================================================
# CALIBRATION CURVE
# ============================================================

def create_calibration_plot(
    y_true,
    probability_dict,
):

    plt.figure(figsize=(8, 7))

    for model_name, probabilities in probability_dict.items():

        fraction_positive, mean_predicted = (
            calibration_curve(
                y_true,
                probabilities,
                n_bins=10,
                strategy="quantile",
            )
        )

        plt.plot(
            mean_predicted,
            fraction_positive,
            marker="o",
            label=model_name,
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")

    plt.title(
        "Probability Calibration Comparison"
    )

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    output_path = (
        RESULTS_FIGURE_DIR
        / "calibration_curve.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nSaved calibration curve:\n{output_path}"
    )


# ============================================================
# THRESHOLD PLOTS
# ============================================================

def create_threshold_plots(
    results,
):

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["Threshold"],
        results["Precision"],
        label="Precision",
    )

    plt.plot(
        results["Threshold"],
        results["Recall"],
        label="Recall",
    )

    plt.plot(
        results["Threshold"],
        results["F1"],
        label="F1",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")

    plt.title(
        "Validation Performance Across Thresholds"
    )

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURE_DIR
        / "validation_threshold_performance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Error tradeoff
    # --------------------------------------------------------

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

    plt.title(
        "Validation False-Negative vs False-Positive Trade-off"
    )

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURE_DIR
        / "validation_error_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Cost
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["Threshold"],
        results["Cost_FN5"],
        label="FN = 5×FP",
    )

    plt.plot(
        results["Threshold"],
        results["Cost_FN10"],
        label="FN = 10×FP",
    )

    plt.plot(
        results["Threshold"],
        results["Cost_FN20"],
        label="FN = 20×FP",
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Illustrative Cost")

    plt.title(
        "Validation Cost Across Decision Thresholds"
    )

    plt.legend()
    plt.grid(alpha=0.25)

    plt.tight_layout()

    plt.savefig(
        RESULTS_FIGURE_DIR
        / "validation_cost_analysis.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def evaluate_on_test(
    model,
    X_test,
    y_test,
    threshold,
    model_name,
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = calculate_metrics(
        y_test,
        probabilities,
        threshold,
    )

    metrics["Model"] = model_name

    metrics["PR_AUC"] = (
        average_precision_score(
            y_test,
            probabilities,
        )
    )

    metrics["ROC_AUC"] = (
        roc_auc_score(
            y_test,
            probabilities,
        )
    )

    metrics["Brier_Score"] = (
        brier_score_loss(
            y_test,
            probabilities,
        )
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "SMARTPREDICT-PdM"
    )
    print(
        "Validation-Based Threshold Optimization + Calibration"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split_data()

    # --------------------------------------------------------
    # Load optimized model
    # --------------------------------------------------------

    model = load_optimized_model()

    # --------------------------------------------------------
    # IMPORTANT:
    # The saved model has already been trained.
    #
    # We use validation probabilities only for threshold
    # selection. Final test data remains untouched.
    # --------------------------------------------------------

    print("\nGenerating validation probabilities...")

    validation_probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    # --------------------------------------------------------
    # Threshold optimization
    # --------------------------------------------------------

    print(
        "\nOptimizing decision threshold on validation set..."
    )

    threshold_results = optimize_thresholds(
        y_val,
        validation_probabilities,
    )

    threshold_results.to_csv(
        RESULTS_TABLE_DIR
        / "validation_threshold_analysis.csv",
        index=False,
    )

    selected_thresholds = select_thresholds(
        threshold_results
    )

    selected_thresholds.to_csv(
        RESULTS_TABLE_DIR
        / "validation_optimal_thresholds.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print threshold results
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("VALIDATION THRESHOLD RESULTS")
    print("=" * 75)

    print(
        selected_thresholds[
            [
                "Criterion",
                "Threshold",
                "Precision",
                "Recall",
                "F1",
                "MCC",
                "FP",
                "FN",
                "Cost_FN5",
                "Cost_FN10",
                "Cost_FN20",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    create_threshold_plots(
        threshold_results
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    probability_dict = calibrate_models(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    calibration_rows = []

    for model_name, probabilities in (
        probability_dict.items()
    ):

        calibration_rows.append(
            calibration_metrics(
                y_val,
                probabilities,
                model_name,
            )
        )

    calibration_results = pd.DataFrame(
        calibration_rows
    )

    calibration_results.to_csv(
        RESULTS_TABLE_DIR
        / "calibration_results.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("CALIBRATION RESULTS")
    print("=" * 75)

    print(
        calibration_results.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    create_calibration_plot(
        y_val,
        probability_dict,
    )

    # --------------------------------------------------------
    # Select final threshold
    #
    # We use the validation maximum-MCC threshold as the
    # primary threshold for the final untouched test.
    # --------------------------------------------------------

    mcc_row = selected_thresholds[
        selected_thresholds["Criterion"]
        == "Maximum MCC"
    ].iloc[0]

    final_threshold = float(
        mcc_row["Threshold"]
    )

    print(
        f"\nFrozen validation threshold: "
        f"{final_threshold:.2f}"
    )

    # --------------------------------------------------------
    # Final untouched test evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating frozen threshold on FINAL TEST SET..."
    )

    final_test = evaluate_on_test(
        model,
        X_test,
        y_test,
        final_threshold,
        "Optimized XGBoost",
    )

    final_test_df = pd.DataFrame(
        [final_test]
    )

    final_test_df.to_csv(
        RESULTS_TABLE_DIR
        / "cost_aware_final_test.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("FINAL TEST RESULT")
    print("=" * 75)

    print(
        final_test_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\nGenerated files:")

    files = [
        "validation_threshold_analysis.csv",
        "validation_optimal_thresholds.csv",
        "calibration_results.csv",
        "cost_aware_final_test.csv",
        "validation_threshold_performance.png",
        "validation_error_tradeoff.png",
        "validation_cost_analysis.png",
        "calibration_curve.png",
    ]

    for file in files:
        print(
            RESULTS_TABLE_DIR / file
            if file.endswith(".csv")
            else RESULTS_FIGURE_DIR / file
        )

    print(
        "\nValidation-based threshold and calibration analysis completed."
    )


if __name__ == "__main__":
    main()