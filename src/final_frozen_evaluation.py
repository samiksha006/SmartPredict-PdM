import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

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

from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR,
)

from preprocessing import build_feature_frame


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

OUTPUT_DIR = (
    RESULTS_TABLE_DIR / "final_frozen"
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
                StandardScaler()
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
# FROZEN OPTIMIZED XGBOOST
# ============================================================

def build_model():

    model = XGBClassifier(

        n_estimators=350,

        max_depth=7,

        learning_rate=0.022323520132149687,

        subsample=0.9474682096161544,

        colsample_bytree=0.9602378472844514,

        min_child_weight=1,

        gamma=0.14439670142040958,

        reg_alpha=0.0006530679464151186,

        reg_lambda=0.11243724759917004,

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
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

        "Threshold":
            threshold,

        "Accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "Precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "ROC_AUC":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "PR_AUC":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                predictions,
            ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SMARTPREDICT-PdM")
    print("FINAL FROZEN MODEL EVALUATION")
    print("=" * 80)


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_data()


    # --------------------------------------------------------
    # 80/20 development-test split
    # --------------------------------------------------------

    X_dev, X_test, y_dev, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )


    # --------------------------------------------------------
    # 80/20 train-validation split
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_dev,
            y_dev,
            test_size=0.20,
            stratify=y_dev,
            random_state=RANDOM_STATE,
        )
    )


    print("\nFinal protocol:")

    print(
        f"Training   : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_val)}"
    )

    print(
        f"Final Test : {len(X_test)}"
    )

    print(
        f"\nTraining failures   : {y_train.sum()}"
    )

    print(
        f"Validation failures : {y_val.sum()}"
    )

    print(
        f"Final test failures : {y_test.sum()}"
    )


    # --------------------------------------------------------
    # Train frozen model
    # --------------------------------------------------------

    print(
        "\nTraining frozen optimized XGBoost..."
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )


    # --------------------------------------------------------
    # Frozen thresholds
    # --------------------------------------------------------

    thresholds = [
        0.50,
        0.20,
        0.11,
        0.05,
    ]


    results = []


    for threshold in thresholds:

        result = evaluate(
            y_test,
            test_probabilities,
            threshold,
        )

        if threshold == 0.50:

            result[
                "Operating_Point"
            ] = "Default reference"

        elif threshold == 0.20:

            result[
                "Operating_Point"
            ] = "1:1 FN:FP"

        elif threshold == 0.11:

            result[
                "Operating_Point"
            ] = "5:1 FN:FP"

        elif threshold == 0.05:

            result[
                "Operating_Point"
            ] = "10:1 / 20:1 FN:FP"


        results.append(
            result
        )


    results_df = pd.DataFrame(
        results
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "final_frozen_test_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL UNTOUCHED TEST RESULTS"
    )

    print(
        "=" * 80
    )


    columns = [

        "Operating_Point",

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


    print(
        results_df[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


    print(
        "\nSaved:"
    )

    print(output_path)


    print(
        "\nFINAL TEST SET WAS NOT USED "
        "FOR THRESHOLD SELECTION."
    )


if __name__ == "__main__":

    main()