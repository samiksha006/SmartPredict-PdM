from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from xgboost import XGBClassifier
from joblib import dump

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    MODELS_DIR,
    RESULTS_TABLE_DIR,
)

from preprocessing import (
    build_feature_frame,
    build_preprocessor,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

FINAL_DIR = RESULTS_TABLE_DIR / "final_experiment"
FINAL_MODEL_DIR = MODELS_DIR / "final_experiment"

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATA
# ============================================================

def load_data():

    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)

    X = build_feature_frame(df)

    if TARGET_COLUMN in X.columns:
        X = X.drop(
            columns=[TARGET_COLUMN]
        )

    y = df[TARGET_COLUMN]

    return X, y


# ============================================================
# SPLIT
# ============================================================

def create_master_split(X, y):

    print("\nCreating master train/validation/test split...")

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_dev,
        y_dev,
        test_size=VALIDATION_SIZE,
        stratify=y_dev,
        random_state=RANDOM_STATE,
    )

    print("\nMASTER SPLIT")

    print(
        f"Training   : {len(X_train)} "
        f"| failures = {y_train.sum()}"
    )

    print(
        f"Validation : {len(X_val)} "
        f"| failures = {y_val.sum()}"
    )

    print(
        f"Final Test : {len(X_test)} "
        f"| failures = {y_test.sum()}"
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
# MODEL DEFINITIONS
# ============================================================

def build_models():

    models = {}

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    models["Logistic Regression"] = (
        LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=RANDOM_STATE,
        )
    )

    models["Decision Tree"] = (
        DecisionTreeClassifier(
            random_state=RANDOM_STATE
        )
    )

    models["KNN"] = (
        KNeighborsClassifier(
            n_neighbors=5
        )
    )

    # --------------------------------------------------------
    # Kernel machines
    # --------------------------------------------------------

    models["Linear SVM"] = SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        random_state=RANDOM_STATE,
    )

    models["Polynomial SVM"] = SVC(
        kernel="poly",
        C=1.0,
        degree=3,
        probability=True,
        random_state=RANDOM_STATE,
    )

    models["RBF SVM"] = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Ensembles
    # --------------------------------------------------------

    models["Random Forest"] = (
        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    models["Extra Trees"] = (
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    )

    models["AdaBoost"] = (
        AdaBoostClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        )
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return models


# ============================================================
# PIPELINE
# ============================================================

def make_pipeline(model):

    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "model",
                model
            ),
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    threshold=0.50,
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "Accuracy": accuracy_score(
            y,
            predictions,
        ),

        "Precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),

        "Recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),

        "F1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),

        "ROC_AUC": roc_auc_score(
            y,
            probabilities,
        ),

        "PR_AUC": average_precision_score(
            y,
            probabilities,
        ),

        "MCC": matthews_corrcoef(
            y,
            predictions,
        ),

        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def find_validation_threshold(
    model,
    X_val,
    y_val,
):

    probabilities = model.predict_proba(
        X_val
    )[:, 1]

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

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predictions,
            labels=[0, 1],
        ).ravel()

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

        cost_5 = (
            fn * 5
            + fp
        )

        cost_10 = (
            fn * 10
            + fp
        )

        cost_20 = (
            fn * 20
            + fp
        )

        rows.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Cost_FN5": cost_5,
            "Cost_FN10": cost_10,
            "Cost_FN20": cost_20,
        })

    results = pd.DataFrame(rows)

    best_mcc = results.loc[
        results["MCC"].idxmax()
    ]

    best_cost = results.loc[
        results["Cost_FN10"].idxmin()
    ]

    return (
        results,
        best_mcc,
        best_cost,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    filename,
):

    path = (
        FINAL_MODEL_DIR
        / filename
    )

    dump(
        model,
        path,
    )

    print(
        f"Saved model: {path}"
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
        "MASTER EXPERIMENT"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Master split
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = create_master_split(
        X,
        y,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = build_models()

    results = []

    # --------------------------------------------------------
    # Train and evaluate models
    # --------------------------------------------------------

    for name, base_model in models.items():

        print(
            "\n" + "-" * 70
        )

        print(
            f"Training: {name}"
        )

        pipeline = make_pipeline(
            base_model
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        # -----------------------------------------------
        # Validation threshold
        # -----------------------------------------------

        (
            threshold_results,
            best_mcc,
            best_cost,
        ) = find_validation_threshold(
            pipeline,
            X_val,
            y_val,
        )

        threshold_results.to_csv(
            FINAL_DIR
            / f"{name.replace(' ', '_').lower()}_thresholds.csv",
            index=False,
        )

        threshold = float(
            best_mcc["Threshold"]
        )

        # -----------------------------------------------
        # Final test using frozen threshold
        # -----------------------------------------------

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test,
            threshold=threshold,
        )

        metrics["Model"] = name
        metrics["Threshold"] = threshold

        results.append(
            metrics
        )

        print(
            f"Threshold : {threshold:.2f}"
        )

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

        # -----------------------------------------------
        # Save model
        # -----------------------------------------------

        filename = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
            + ".joblib"
        )

        save_model(
            pipeline,
            filename,
        )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

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
        FINAL_DIR
        / "master_test_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 75)
    print(
        "FINAL MASTER TEST RESULTS"
    )
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------------
    # Save split metadata
    # --------------------------------------------------------

    metadata = {
        "random_state": RANDOM_STATE,
        "total_samples": len(X),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "training_failures": int(y_train.sum()),
        "validation_failures": int(y_val.sum()),
        "test_failures": int(y_test.sum()),
        "target": TARGET_COLUMN,
        "features": list(X.columns),
    }

    metadata_path = (
        FINAL_DIR
        / "master_experiment_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )

    print(
        f"\nSaved results:\n{output_path}"
    )

    print(
        f"Saved metadata:\n{metadata_path}"
    )

    print(
        "\nMASTER EXPERIMENT COMPLETED."
    )


if __name__ == "__main__":
    main()