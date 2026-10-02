import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

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

OUTPUT_DIR = (
    RESULTS_TABLE_DIR / "cost_sensitive"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD DATA
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
# METRICS
# ============================================================

def calculate_metrics(
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

        "Accuracy":
            (tp + tn)
            /
            (tp + tn + fp + fn),

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

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,

        "Cost":
            cost,
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def search_thresholds(
    y_true,
    probabilities,
    fn_cost,
    fp_cost,
):

    thresholds = np.arange(
        0.05,
        0.951,
        0.01,
    )

    rows = []

    for threshold in thresholds:

        row = calculate_metrics(
            y_true,
            probabilities,
            threshold,
            fn_cost,
            fp_cost,
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "SMARTPREDICT-PdM"
    )

    print(
        "COST-SENSITIVE THRESHOLD ANALYSIS"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_data()


    # --------------------------------------------------------
    # First split:
    # Development / Final Test
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
    # Second split:
    # Training / Validation
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
        f"Test failures       : {y_test.sum()}"
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    print(
        "\nTraining optimized XGBoost..."
    )


    model = build_model()

    model.fit(
        X_train,
        y_train,
    )


    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )


    # ========================================================
    # COST SCENARIOS
    # ========================================================

    cost_ratios = [
        1,
        5,
        10,
        20,
    ]


    all_results = []

    selected_thresholds = []


    for ratio in cost_ratios:

        print(
            "\n" + "-" * 70
        )

        print(
            f"FN COST = {ratio} × FP COST"
        )

        print(
            "-" * 70
        )


        validation_results = (
            search_thresholds(

                y_val,

                val_probabilities,

                fn_cost=ratio,

                fp_cost=1,
            )
        )


        best_row = (
            validation_results
            .sort_values(
                "Cost",
                ascending=True,
            )
            .iloc[0]
        )


        threshold = float(
            best_row["Threshold"]
        )


        selected_thresholds.append({

            "FN_Cost_Ratio":
                ratio,

            "Selected_Threshold":
                threshold,

            "Validation_Cost":
                best_row["Cost"],

            "Validation_Precision":
                best_row["Precision"],

            "Validation_Recall":
                best_row["Recall"],

            "Validation_F1":
                best_row["F1"],

            "Validation_MCC":
                best_row["MCC"],

            "Validation_FP":
                best_row["FP"],

            "Validation_FN":
                best_row["FN"],

        })


        # ----------------------------------------------------
        # Apply threshold ONCE to final test
        # ----------------------------------------------------

        test_probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )


        test_result = (
            calculate_metrics(

                y_test,

                test_probabilities,

                threshold,

                fn_cost=ratio,

                fp_cost=1,
            )
        )


        test_result[
            "FN_Cost_Ratio"
        ] = ratio


        test_result[
            "Validation_Selected_Threshold"
        ] = threshold


        all_results.append(
            test_result
        )


        print(
            f"Selected threshold : "
            f"{threshold:.2f}"
        )

        print(
            f"Validation cost : "
            f"{best_row['Cost']:.0f}"
        )

        print(
            f"Test precision : "
            f"{test_result['Precision']:.4f}"
        )

        print(
            f"Test recall : "
            f"{test_result['Recall']:.4f}"
        )

        print(
            f"Test F1 : "
            f"{test_result['F1']:.4f}"
        )

        print(
            f"Test MCC : "
            f"{test_result['MCC']:.4f}"
        )

        print(
            f"Test FP : "
            f"{test_result['FP']}"
        )

        print(
            f"Test FN : "
            f"{test_result['FN']}"
        )


    # ========================================================
    # SAVE VALIDATION THRESHOLDS
    # ========================================================

    threshold_df = pd.DataFrame(
        selected_thresholds
    )


    threshold_path = (
        OUTPUT_DIR
        / "selected_thresholds.csv"
    )


    threshold_df.to_csv(
        threshold_path,
        index=False,
    )


    # ========================================================
    # SAVE TEST RESULTS
    # ========================================================

    test_df = pd.DataFrame(
        all_results
    )


    test_path = (
        OUTPUT_DIR
        / "cost_sensitive_test_results.csv"
    )


    test_df.to_csv(
        test_path,
        index=False,
    )


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "COST-SENSITIVE TEST SUMMARY"
    )

    print(
        "=" * 80
    )


    columns = [

        "FN_Cost_Ratio",

        "Validation_Selected_Threshold",

        "Precision",

        "Recall",

        "F1",

        "MCC",

        "PR_AUC",

        "FP",

        "FN",

        "Cost",
    ]


    print(
        test_df[
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

    print(
        threshold_path
    )

    print(
        test_path
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "Thresholds were selected using validation data."
    )

    print(
        "The final test set was evaluated only "
        "after threshold selection."
    )


if __name__ == "__main__":

    main()