from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_validate,
)

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR,
)

from preprocessing import build_feature_frame


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5
N_REPEATS = 3

OUTPUT_DIR = (
    RESULTS_TABLE_DIR / "repeated_cv"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor():

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

    numeric_pipeline = Pipeline(
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

    categorical_pipeline = Pipeline(
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
# XGBOOST MODELS
# ============================================================

def create_xgboost(
    optimized=False,
    class_weight=False,
):

    if optimized:

        params = {
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

        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }

    if class_weight:

        params["scale_pos_weight"] = (
            9661 / 339
        )

    return XGBClassifier(
        **params,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ============================================================
# PIPELINE BUILDER
# ============================================================

def build_model(stage):

    preprocessor = (
        build_preprocessor()
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
# SCORING
# ============================================================

def create_scoring():

    return {
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),

        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),

        "mcc": make_scorer(
            matthews_corrcoef,
        ),

        "roc_auc": "roc_auc",

        "pr_auc": "average_precision",
    }


# ============================================================
# RUN CROSS VALIDATION
# ============================================================

def run_cv(
    stage,
    X_dev,
    y_dev,
    cv,
):

    print(
        "\n" + "-" * 75
    )

    print(
        f"Running repeated CV: {stage}"
    )

    model = build_model(
        stage
    )

    scoring = create_scoring()

    results = cross_validate(
        model,
        X_dev,
        y_dev,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False,
    )

    rows = []

    for fold_index in range(
        len(results["test_pr_auc"])
    ):

        rows.append({
            "Stage": stage,
            "Fold": fold_index + 1,

            "Precision": results[
                "test_precision"
            ][fold_index],

            "Recall": results[
                "test_recall"
            ][fold_index],

            "F1": results[
                "test_f1"
            ][fold_index],

            "MCC": results[
                "test_mcc"
            ][fold_index],

            "ROC_AUC": results[
                "test_roc_auc"
            ][fold_index],

            "PR_AUC": results[
                "test_pr_auc"
            ][fold_index],

            "Fit_Time": results[
                "fit_time"
            ][fold_index],
        })

    fold_results = pd.DataFrame(
        rows
    )

    return fold_results


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(
    fold_results
):

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "ROC_AUC",
        "PR_AUC",
    ]

    summaries = []

    for stage, group in (
        fold_results.groupby("Stage")
    ):

        row = {
            "Stage": stage,
        }

        for metric in metrics:

            row[
                f"{metric}_Mean"
            ] = group[metric].mean()

            row[
                f"{metric}_Std"
            ] = group[metric].std(
                ddof=1
            )

        row[
            "Fit_Time_Mean"
        ] = group["Fit_Time"].mean()

        summaries.append(
            row
        )

    return pd.DataFrame(
        summaries
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "SMARTPREDICT-PdM"
    )
    print(
        "REPEATED STRATIFIED CROSS-VALIDATION"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Reserve final test set
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    print("\nData allocation:")

    print(
        f"Development data : {len(X_dev)}"
    )

    print(
        f"Final test data  : {len(X_test)}"
    )

    print(
        f"\nDevelopment failures : "
        f"{y_dev.sum()}"
    )

    print(
        f"Final test failures  : "
        f"{y_test.sum()}"
    )

    # --------------------------------------------------------
    # Repeated stratified CV
    # --------------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nCross-validation:"
    )

    print(
        f"Folds       : {N_SPLITS}"
    )

    print(
        f"Repetitions : {N_REPEATS}"
    )

    print(
        f"Total fits per configuration : "
        f"{N_SPLITS * N_REPEATS}"
    )

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    stages = [
        "baseline",
        "optimized",
        "class_weight",
        "smote",
        "smote_tomek",
    ]

    all_fold_results = []

    for stage in stages:

        fold_results = run_cv(
            stage,
            X_dev,
            y_dev,
            cv,
        )

        all_fold_results.append(
            fold_results
        )

        print(
            f"\n{stage.upper()}:"
        )

        print(
            f"PR-AUC: "
            f"{fold_results['PR_AUC'].mean():.4f} "
            f"+/- "
            f"{fold_results['PR_AUC'].std():.4f}"
        )

        print(
            f"Recall: "
            f"{fold_results['Recall'].mean():.4f} "
            f"+/- "
            f"{fold_results['Recall'].std():.4f}"
        )

        print(
            f"F1: "
            f"{fold_results['F1'].mean():.4f} "
            f"+/- "
            f"{fold_results['F1'].std():.4f}"
        )

        print(
            f"MCC: "
            f"{fold_results['MCC'].mean():.4f} "
            f"+/- "
            f"{fold_results['MCC'].std():.4f}"
        )

    # --------------------------------------------------------
    # Combine fold results
    # --------------------------------------------------------

    fold_df = pd.concat(
        all_fold_results,
        ignore_index=True,
    )

    fold_path = (
        OUTPUT_DIR
        / "repeated_cv_fold_results.csv"
    )

    fold_df.to_csv(
        fold_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_df = summarize_results(
        fold_df
    )

    summary_path = (
        OUTPUT_DIR
        / "repeated_cv_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print(
        "REPEATED CV SUMMARY"
    )
    print("=" * 75)

    display_columns = [
        "Stage",
        "PR_AUC_Mean",
        "PR_AUC_Std",
        "Recall_Mean",
        "Recall_Std",
        "F1_Mean",
        "F1_Std",
        "MCC_Mean",
        "MCC_Std",
        "ROC_AUC_Mean",
        "ROC_AUC_Std",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nSaved fold results:"
    )

    print(
        fold_path
    )

    print(
        "\nSaved summary:"
    )

    print(
        summary_path
    )

    print(
        "\nREPEATED CROSS-VALIDATION COMPLETED."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The final test set was reserved and "
        "was not used during cross-validation."
    )


if __name__ == "__main__":
    main()