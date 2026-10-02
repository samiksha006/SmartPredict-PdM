"""
SmartPredict-PdM
Stage 9: SHAP Explainability
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_TABLE_DIR,
    RESULTS_FIGURE_DIR,
)

from preprocessing import (
    build_feature_frame,
    build_preprocessor,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_SHAP_SAMPLES = 1000


# ============================================================
# OPTUNA BEST PARAMETERS
# ============================================================

OPTIMIZED_PARAMS = {
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


# ============================================================
# CREATE RESULT DIRECTORIES
# ============================================================

RESULTS_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("STAGE 9: SHAP EXPLAINABILITY")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# BUILD FEATURES
# ============================================================

X = build_feature_frame(df)

if TARGET_COLUMN in X.columns:
    X = X.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN]

print("\nFeatures used:")
print(list(X.columns))

print(f"\nFailure rate: {y.mean() * 100:.2f}%")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 70)
print("TEST SET")
print("=" * 70)

print(f"Training samples : {len(X_train)}")
print(f"Test samples     : {len(X_test)}")
print(f"Test failures    : {int(y_test.sum())}")


# ============================================================
# BUILD OPTIMIZED XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING OPTIMIZED XGBOOST")
print("=" * 70)

preprocessor = build_preprocessor()

xgb_model = XGBClassifier(
    **OPTIMIZED_PARAMS,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)

pipeline.fit(
    X_train,
    y_train
)

print("Optimized XGBoost training completed.")


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

print("\nTransforming test data for SHAP...")

X_test_transformed = pipeline.named_steps[
    "preprocessor"
].transform(X_test)


# ============================================================
# CONVERT SPARSE MATRIX IF NECESSARY
# ============================================================

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

X_test_transformed = np.asarray(
    X_test_transformed
)


# ============================================================
# GET FEATURE NAMES
# ============================================================

try:
    feature_names = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

except Exception:
    feature_names = np.array(
        [
            f"Feature_{i}"
            for i in range(
                X_test_transformed.shape[1]
            )
        ]
    )

feature_names = np.array(
    [str(name) for name in feature_names]
)

print(
    f"Transformed feature count: "
    f"{len(feature_names)}"
)


# ============================================================
# SELECT SHAP SAMPLE
# ============================================================

if (
    MAX_SHAP_SAMPLES is not None
    and len(X_test_transformed) > MAX_SHAP_SAMPLES
):

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        len(X_test_transformed),
        size=MAX_SHAP_SAMPLES,
        replace=False
    )

    X_shap = X_test_transformed[
        sample_indices
    ]

else:
    X_shap = X_test_transformed


print(
    f"SHAP samples: {len(X_shap)}"
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SHAP VALUES")
print("=" * 70)

model = pipeline.named_steps["model"]

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_shap
)


# ============================================================
# HANDLE SHAP OUTPUT FORMAT
# ============================================================

if isinstance(shap_values, list):
    shap_values_array = shap_values[1]
else:
    shap_values_array = shap_values

shap_values_array = np.asarray(
    shap_values_array
)

print(
    f"SHAP matrix shape: "
    f"{shap_values_array.shape}"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values_array),
    axis=0
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

importance_df["Rank"] = (
    np.arange(len(importance_df)) + 1
)

importance_df = importance_df[
    [
        "Rank",
        "Feature",
        "Mean_Absolute_SHAP"
    ]
]


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = (
    RESULTS_TABLE_DIR /
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP SHAP FEATURES")
print("=" * 70)

print(
    importance_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)


# ============================================================
# SHAP BAR PLOT
# ============================================================

print("\nGenerating SHAP bar plot...")

shap.summary_plot(
    shap_values_array,
    X_shap,
    feature_names=feature_names,
    plot_type="bar",
    show=False,
    max_display=15
)

plt.title(
    "SHAP Feature Importance - Optimized XGBoost"
)

plt.tight_layout()

bar_path = (
    RESULTS_FIGURE_DIR /
    "shap_bar.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print("Generating SHAP summary plot...")

shap.summary_plot(
    shap_values_array,
    X_shap,
    feature_names=feature_names,
    show=False,
    max_display=15
)

plt.title(
    "SHAP Summary - Optimized XGBoost"
)

plt.tight_layout()

summary_path = (
    RESULTS_FIGURE_DIR /
    "shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP WATERFALL PLOT
# ============================================================

print("Generating SHAP waterfall plot...")

sample_position = int(
    np.argmax(
        np.sum(
            np.abs(shap_values_array),
            axis=1
        )
    )
)

try:

    base_value = explainer.expected_value

    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    explanation = shap.Explanation(
        values=shap_values_array[
            sample_position
        ],
        base_values=base_value,
        data=X_shap[
            sample_position
        ],
        feature_names=feature_names
    )

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.tight_layout()

    waterfall_path = (
        RESULTS_FIGURE_DIR /
        "shap_waterfall.png"
    )

    plt.savefig(
        waterfall_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Waterfall plot generated.")

except Exception as e:

    waterfall_path = None

    print(
        f"Warning: Waterfall plot could not "
        f"be generated: {e}"
    )


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_values_df = pd.DataFrame(
    shap_values_array,
    columns=feature_names
)

shap_values_path = (
    RESULTS_TABLE_DIR /
    "shap_values.csv"
)

shap_values_df.to_csv(
    shap_values_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("\nFeature importance:")
print(importance_path)

print("\nSHAP values:")
print(shap_values_path)

print("\nFigures:")
print(bar_path)
print(summary_path)

if waterfall_path is not None:
    print(waterfall_path)

print("\n" + "=" * 70)
print("STAGE 9 COMPLETED")
print("=" * 70)
