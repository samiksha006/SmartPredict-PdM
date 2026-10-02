"""
SmartPredict-PdM
Stage 7 — Final Held-Out Test Evaluation
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
    MODELS_DIR
)

from preprocessing import (
    build_preprocessor,
    build_feature_frame
)

warnings.filterwarnings("ignore")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("FINAL HELD-OUT TEST EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

X = build_feature_frame(df)

if TARGET_COLUMN in X.columns:
    X = X.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

print(
    f"\nDataset shape: {df.shape}"
)

print(
    f"Features: {list(X.columns)}"
)

print(
    f"Failure rate: {y.mean() * 100:.2f}%"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    stratify=y,

    random_state=42
)


print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Test samples: {len(X_test)}"
)

print(
    f"Test failures: {y_test.sum()}"
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = build_preprocessor()


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "XGBoost Baseline":

        XGBClassifier(

            n_estimators=300,

            max_depth=4,

            learning_rate=0.05,

            subsample=0.9,

            colsample_bytree=0.9,

            eval_metric="logloss",

            random_state=42,

            n_jobs=-1
        ),


    "Random Forest":

        __import__(
            "sklearn.ensemble",
            fromlist=["RandomForestClassifier"]
        ).RandomForestClassifier(

            n_estimators=300,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        ),


    "Extra Trees":

        __import__(
            "sklearn.ensemble",
            fromlist=["ExtraTreesClassifier"]
        ).ExtraTreesClassifier(

            n_estimators=300,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1
        ),


    "Optimized XGBoost":

        XGBClassifier(

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

            random_state=42,

            n_jobs=-1
        )
}


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TRAIN + EVALUATE
# ============================================================

for name, model in models.items():

    print("\n" + "=" * 70)

    print(
        f"Evaluating: {name}"
    )

    print("=" * 70)


    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ])


    pipeline.fit(
        X_train,
        y_train
    )


    y_pred = pipeline.predict(
        X_test
    )


    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )


    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"MCC       : {mcc:.4f}"
    )

    print(
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )


    results.append({

        "Model": name,

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

        "TP": tp
    })


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_final.joblib"
    )

    joblib.dump(

        pipeline,

        os.path.join(
            MODELS_DIR,
            filename
        )
    )


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal",
            "Failure"
        ]
    ).plot(
        ax=ax
    )

    ax.set_title(
        f"{name} — Confusion Matrix"
    )

    plt.tight_layout()

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    plt.savefig(

        os.path.join(

            RESULTS_FIGURE_DIR,

            f"{safe_name}_confusion_matrix.png"
        ),

        dpi=300
    )

    plt.close()


# ============================================================
# SMOTE-TOMEK FINAL MODEL
# ============================================================

print("\n" + "=" * 70)

print(
    "Evaluating: XGBoost + SMOTE-Tomek"
)

print("=" * 70)


smote_model = XGBClassifier(

    n_estimators=300,

    max_depth=4,

    learning_rate=0.05,

    subsample=0.9,

    colsample_bytree=0.9,

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)


smote_pipeline = ImbPipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "smote_tomek",
        SMOTETomek(
            random_state=42
        )
    ),

    (
        "model",
        smote_model
    )
])


smote_pipeline.fit(
    X_train,
    y_train
)


y_pred = smote_pipeline.predict(
    X_test
)

y_prob = smote_pipeline.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

pr_auc = average_precision_score(
    y_test,
    y_prob
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print(
    f"PR-AUC    : {pr_auc:.4f}"
)

print(
    f"MCC       : {mcc:.4f}"
)

print(
    f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
)


results.append({

    "Model": "XGBoost + SMOTE-Tomek",

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

    "TP": tp
})


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    "PR_AUC",
    ascending=False
)


print("\n")

print("=" * 70)

print("FINAL TEST RESULTS")

print("=" * 70)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(

    RESULTS_TABLE_DIR,

    "final_test_results.csv"
)


results_df.to_csv(

    output_path,

    index=False
)


print(
    f"\nSaved final results:"
)

print(
    output_path
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)

print("STAGE 7 COMPLETED")

print("=" * 70)