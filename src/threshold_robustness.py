import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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
    RESULTS_TABLE_DIR / "threshold_robustness"
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
# OPTIMIZED XGBOOST
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
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    y_true,
    probabilities,
    threshold,
    fn_cost,
    fp_cost,
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

    cost = (
        fn * fn_cost
        +
        fp * fp_cost
    )

    return {

        "Threshold": threshold,

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

        "MCC":
            matthews_corrcoef(
                y_true,
                predictions,
            ),

        "TN": tn,

        "FP": fp,

        "FN": fn,

        "TP": tp,

        "Cost": cost,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SMARTPREDICT-PdM")
    print("THRESHOLD ROBUSTNESS ANALYSIS")
    print("=" * 80)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_data()


    # --------------------------------------------------------
    # Development / final test
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
    # Training / validation
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


    print("\nData allocation:")
    print(f"Training   : {len(X_train)}")
    print(f"Validation : {len(X_val)}")
    print(f"Final Test : {len(X_test)}")

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
    # Train model
    # --------------------------------------------------------

    print(
        "\nTraining optimized XGBoost..."
    )

    model = build_model()

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )


    # --------------------------------------------------------
    # Fine threshold grid
    # --------------------------------------------------------

    thresholds = np.arange(
        0.01,
        0.201,
        0.01,
    )


    cost_ratios = [
        1,
        5,
        10,
        20,
    ]


    all_rows = []


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    for cost_ratio in cost_ratios:

        for threshold in thresholds:

            result = evaluate_threshold(

                y_val,

                val_probabilities,

                threshold,

                fn_cost=cost_ratio,

                fp_cost=1,
            )

            result[
                "FN_Cost_Ratio"
            ] = cost_ratio

            all_rows.append(
                result
            )


    results_df = pd.DataFrame(
        all_rows
    )


    # --------------------------------------------------------
    # Save complete analysis
    # --------------------------------------------------------

    full_path = (
        OUTPUT_DIR
        / "threshold_robustness_all.csv"
    )

    results_df.to_csv(
        full_path,
        index=False,
    )


    # --------------------------------------------------------
    # Find minimum-cost threshold
    # --------------------------------------------------------

    selected_rows = []


    for cost_ratio in cost_ratios:

        subset = results_df[
            results_df[
                "FN_Cost_Ratio"
            ]
            == cost_ratio
        ].copy()

        # Find minimum cost.
        min_cost = subset[
            "Cost"
        ].min()

        candidates = subset[
            subset["Cost"]
            == min_cost
        ].copy()

        # If several thresholds have
        # identical cost, prefer the
        # higher threshold because it
        # avoids unnecessary alerts.
        best = candidates.sort_values(
            "Threshold",
            ascending=False,
        ).iloc[0]

        selected_rows.append({

            "FN_Cost_Ratio":
                cost_ratio,

            "Selected_Threshold":
                best["Threshold"],

            "Validation_Cost":
                best["Cost"],

            "Precision":
                best["Precision"],

            "Recall":
                best["Recall"],

            "F1":
                best["F1"],

            "MCC":
                best["MCC"],

            "FP":
                best["FP"],

            "FN":
                best["FN"],

            "TP":
                best["TP"],

            "TN":
                best["TN"],
        })


    selected_df = pd.DataFrame(
        selected_rows
    )


    selected_path = (
        OUTPUT_DIR
        / "threshold_robustness_selected.csv"
    )


    selected_df.to_csv(
        selected_path,
        index=False,
    )


    # --------------------------------------------------------
    # Print threshold table
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION THRESHOLD SELECTION"
    )

    print(
        "=" * 80
    )


    print(
        selected_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


    # --------------------------------------------------------
    # Print low-threshold details
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "LOW-THRESHOLD ANALYSIS"
    )

    print(
        "=" * 80
    )


    for cost_ratio in cost_ratios:

        subset = results_df[
            results_df[
                "FN_Cost_Ratio"
            ]
            == cost_ratio
        ]

        print(
            f"\nFN COST = {cost_ratio} × FP COST"
        )

        print(
            subset[
                [
                    "Threshold",
                    "Precision",
                    "Recall",
                    "F1",
                    "MCC",
                    "FP",
                    "FN",
                    "Cost",
                ]
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )


    print(
        "\n" + "=" * 80
    )

    print(
        "FILES SAVED"
    )

    print(
        "=" * 80
    )

    print(full_path)

    print(selected_path)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Threshold analysis used validation data only."
    )

    print(
        "The final test set was not used "
        "for threshold selection."
    )


if __name__ == "__main__":

    main()
