from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

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

TEST_SIZE = 0.20

OUTPUT_DIR = (
    RESULTS_TABLE_DIR / "clean_ensemble"
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
# MODEL DEFINITIONS
# ============================================================

def build_xgboost():

    return XGBClassifier(

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


def build_random_forest():

    return RandomForestClassifier(

        n_estimators=400,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        class_weight=None,

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )


def build_extra_trees():

    return ExtraTreesClassifier(

        n_estimators=400,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        class_weight=None,

        random_state=RANDOM_STATE,

        n_jobs=-1,
    )


def build_rbf_svm():

    return SVC(

        C=10,

        kernel="rbf",

        gamma="scale",

        probability=True,

        random_state=RANDOM_STATE,
    )


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(model):

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
# INDIVIDUAL MODELS
# ============================================================

def build_individual_models():

    return {

        "Random Forest":
            build_pipeline(
                build_random_forest()
            ),

        "Extra Trees":
            build_pipeline(
                build_extra_trees()
            ),

        "RBF SVM":
            build_pipeline(
                build_rbf_svm()
            ),

        "Optimized XGBoost":
            build_pipeline(
                build_xgboost()
            ),
    }


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold=0.5,
):

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    return {

        "Accuracy":
            accuracy_score(
                y,
                predictions,
            ),

        "Precision":
            precision_score(
                y,
                predictions,
                zero_division=0,
            ),

        "Recall":
            recall_score(
                y,
                predictions,
                zero_division=0,
            ),

        "F1":
            f1_score(
                y,
                predictions,
                zero_division=0,
            ),

        "ROC_AUC":
            roc_auc_score(
                y,
                probabilities,
            ),

        "PR_AUC":
            average_precision_score(
                y,
                probabilities,
            ),

        "MCC":
            matthews_corrcoef(
                y,
                predictions,
            ),

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "SMARTPREDICT-PdM"
    )

    print(
        "CLEAN VOTING + STACKING ENSEMBLE EXPERIMENT"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_data()


    # --------------------------------------------------------
    # Development / Final Test Split
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


    print("\nDataset split:")

    print(
        f"Development : {len(X_dev)}"
    )

    print(
        f"Final Test  : {len(X_test)}"
    )

    print(
        f"Development failures : "
        f"{y_dev.sum()}"
    )

    print(
        f"Final test failures : "
        f"{y_test.sum()}"
    )


    # --------------------------------------------------------
    # Individual Models
    # --------------------------------------------------------

    individual_models = (
        build_individual_models()
    )


    fitted_models = {}


    results = []


    # --------------------------------------------------------
    # Train Individual Models
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING INDIVIDUAL MODELS"
    )

    print(
        "=" * 80
    )


    for name, model in (
        individual_models.items()
    ):

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            X_dev,
            y_dev,
        )

        fitted_models[name] = model

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        metrics[
            "Model"
        ] = name

        metrics[
            "Threshold"
        ] = 0.5

        results.append(
            metrics
        )

        print(
            f"{name} completed."
        )

        print(
            f"PR-AUC : "
            f"{metrics['PR_AUC']:.4f}"
        )

        print(
            f"Recall : "
            f"{metrics['Recall']:.4f}"
        )

        print(
            f"F1     : "
            f"{metrics['F1']:.4f}"
        )

        print(
            f"MCC    : "
            f"{metrics['MCC']:.4f}"
        )


    # ========================================================
    # VOTING ENSEMBLE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING SOFT VOTING ENSEMBLE"
    )

    print(
        "=" * 80
    )


    voting_estimators = [

        (
            "rf",
            build_pipeline(
                build_random_forest()
            ),
        ),

        (
            "et",
            build_pipeline(
                build_extra_trees()
            ),
        ),

        (
            "svm",
            build_pipeline(
                build_rbf_svm()
            ),
        ),

        (
            "xgb",
            build_pipeline(
                build_xgboost()
            ),
        ),
    ]


    voting = VotingClassifier(

        estimators=voting_estimators,

        voting="soft",

        weights=[
            1,
            1,
            1,
            2,
        ],

        n_jobs=1,
    )


    print(
        "\nFitting Voting Ensemble..."
    )


    voting.fit(
        X_dev,
        y_dev,
    )


    voting_metrics = evaluate_model(
        voting,
        X_test,
        y_test,
    )


    voting_metrics[
        "Model"
    ] = "Soft Voting"

    voting_metrics[
        "Threshold"
    ] = 0.5


    results.append(
        voting_metrics
    )


    print(
        "\nVoting completed."
    )

    print(
        f"PR-AUC : "
        f"{voting_metrics['PR_AUC']:.4f}"
    )

    print(
        f"Recall : "
        f"{voting_metrics['Recall']:.4f}"
    )

    print(
        f"F1     : "
        f"{voting_metrics['F1']:.4f}"
    )

    print(
        f"MCC    : "
        f"{voting_metrics['MCC']:.4f}"
    )


    # ========================================================
    # STACKING ENSEMBLE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING STACKING ENSEMBLE"
    )

    print(
        "=" * 80
    )


    stacking_estimators = [

        (
            "rf",
            build_pipeline(
                build_random_forest()
            ),
        ),

        (
            "et",
            build_pipeline(
                build_extra_trees()
            ),
        ),

        (
            "svm",
            build_pipeline(
                build_rbf_svm()
            ),
        ),

        (
            "xgb",
            build_pipeline(
                build_xgboost()
            ),
        ),
    ]


    stacking = StackingClassifier(

        estimators=stacking_estimators,

        final_estimator=
            LogisticRegression(
                max_iter=2000,
                class_weight=None,
                random_state=RANDOM_STATE,
            ),

        stack_method="predict_proba",

        cv=5,

        n_jobs=1,
    )


    print(
        "\nFitting Stacking Ensemble..."
    )


    stacking.fit(
        X_dev,
        y_dev,
    )


    stacking_metrics = evaluate_model(
        stacking,
        X_test,
        y_test,
    )


    stacking_metrics[
        "Model"
    ] = "Stacking"

    stacking_metrics[
        "Threshold"
    ] = 0.5


    results.append(
        stacking_metrics
    )


    print(
        "\nStacking completed."
    )

    print(
        f"PR-AUC : "
        f"{stacking_metrics['PR_AUC']:.4f}"
    )

    print(
        f"Recall : "
        f"{stacking_metrics['Recall']:.4f}"
    )

    print(
        f"F1     : "
        f"{stacking_metrics['F1']:.4f}"
    )

    print(
        f"MCC    : "
        f"{stacking_metrics['MCC']:.4f}"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    column_order = [

        "Model",
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


    results_df = results_df[
        column_order
    ]


    output_path = (
        OUTPUT_DIR
        / "clean_ensemble_results.csv"
    )


    results_df.to_csv(
        output_path,
        index=False,
    )


    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "CLEAN ENSEMBLE TEST RESULTS"
    )

    print(
        "=" * 80
    )


    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


    print(
        "\nSaved:"
    )

    print(
        output_path
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "This experiment uses the same clean "
        "80/20 development-test split."
    )

    print(
        "The final test set was not used "
        "for model fitting."
    )


if __name__ == "__main__":

    main()