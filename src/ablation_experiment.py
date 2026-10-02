from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from config import (
    DATA_PATH,
    TARGET_COLUMN,
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

ABlation_DIR = (
    RESULTS_TABLE_DIR / "ablation"
)

ABlation_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR = (
    RESULTS_FIGURE_DIR / "ablation"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    X = build_feature_frame(df)

    if TARGET_COLUMN in X.columns:
        X = X.drop(
            columns=[TARGET_COLUMN]
        )

    y = df[TARGET_COLUMN]

    return X, y


def create_split(X, y):

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=VALIDATION_SIZE,
            stratify=y_dev,
            random_state=RANDOM_STATE,
        )
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def build_local_preprocessor():

    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    categorical_features = [
        "Type"
    ]

    numeric_pipeline = SkPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = SkPipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


# ============================================================
# BASE XGBOOST
# ============================================================

def create_xgboost(
    optimized=False,
    class_weight=False,
):

    if optimized:

        model_params = {
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

    else:

        model_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }

    if class_weight:

        model_params["scale_pos_weight"] = (
            9661 / 339
        )

    model = XGBClassifier(
        **model_params,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return model


# ============================================================
# PIPELINES
# ============================================================

def build_pipeline(
    stage
):

    preprocessor = (
        build_local_preprocessor()
    )

    if stage == "baseline":

        model = create_xgboost()

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    if stage == "optimized":

        model = create_xgboost(
            optimized=True
        )

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    if stage == "class_weight":

        model = create_xgboost(
            optimized=True,
            class_weight=True,
        )

        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    if stage == "smote":

        model = create_xgboost(
            optimized=True
        )

        return ImbPipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "smote",
                    SMOTE(
                        random_state=RANDOM_STATE
                    ),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    if stage == "smote_tomek":

        model = create_xgboost(
            optimized=True
        )

        return ImbPipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "smote_tomek",
                    SMOTETomek(
                        random_state=RANDOM_STATE
                    ),
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

    raise ValueError(
        f"Unknown stage: {stage}"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    return {
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities,
        ),

        "PR_AUC": average_precision_score(
            y_true,
            probabilities,
        ),

        "MCC": matthews_corrcoef(
            y_true,
            predictions,
        ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# VALIDATION THRESHOLD
# ============================================================

def find_threshold(
    model,
    X_val,
    y_val,
):

    probabilities = (
        model.predict_proba(X_val)[:, 1]
    )

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_val,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        mcc = matthews_corrcoef(
            y_val,
            predictions,
        )

        cost = (
            fn * 10
            + fp
        )

        rows.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "TN": tn,
            "Cost": cost,
        })

    results = pd.DataFrame(
        rows
    )

    # Primary threshold:
    # maximum MCC
    best_mcc = results.loc[
        results["MCC"].idxmax()
    ]

    return (
        float(best_mcc["Threshold"]),
        results,
    )


# ============================================================
# RUN ONE ABLATION
# ============================================================

def run_stage(
    stage,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"ABLATION STAGE: {stage}"
    )

    model = build_pipeline(
        stage
    )

    print(
        "Training model..."
    )

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    threshold, threshold_results = (
        find_threshold(
            model,
            X_val,
            y_val,
        )
    )

    threshold_results.to_csv(
        ABlation_DIR
        / f"{stage}_thresholds.csv",
        index=False,
    )

    print(
        f"Selected threshold: {threshold:.2f}"
    )

    # --------------------------------------------------------
    # Final test
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(X_test)[:, 1]
    )

    metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    metrics["Stage"] = stage
    metrics["Threshold"] = threshold

    print(
        f"Precision : {metrics['Precision']:.4f}"
    )

    print(
        f"Recall    : {metrics['Recall']:.4f}"
    )

    print(
        f"F1        : {metrics['F1']:.4f}"
    )

    print(
        f"PR-AUC    : {metrics['PR_AUC']:.4f}"
    )

    print(
        f"MCC       : {metrics['MCC']:.4f}"
    )

    print(
        f"FP        : {metrics['FP']}"
    )

    print(
        f"FN        : {metrics['FN']}"
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
        "XGBOOST ABLATION STUDY"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    X, y = load_data()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = create_split(
        X,
        y,
    )

    print("\nDataset split:")

    print(
        f"Training   : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_val)}"
    )

    print(
        f"Final Test : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Ablation stages
    # --------------------------------------------------------

    stages = [
        "baseline",
        "optimized",
        "class_weight",
        "smote",
        "smote_tomek",
    ]

    all_results = []

    for stage in stages:

        result = run_stage(
            stage,
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_df = results_df[
        [
            "Stage",
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC",
            "MCC",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ]

    output_path = (
        ABlation_DIR
        / "ablation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print(
        "ABLATION RESULTS"
    )
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        f"\nSaved:\n{output_path}"
    )

    print(
        "\nABLATION STUDY COMPLETED."
    )


if __name__ == "__main__":
    main()