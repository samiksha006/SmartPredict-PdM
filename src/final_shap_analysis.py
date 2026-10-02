# src/final_shap_analysis.py

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ================================================================
# PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    RESULTS_DIR,
)

from src.preprocessing import (
    build_feature_frame,
    build_preprocessor,
)


# ================================================================
# OUTPUT DIRECTORIES
# ================================================================

SHAP_DIR = RESULTS_DIR / "final_shap_analysis"
FIGURE_DIR = SHAP_DIR / "figures"

SHAP_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# CONFIGURATION
# ================================================================

RANDOM_STATE = 42

# Same optimized parameters used throughout the final experiment
XGB_PARAMS = {
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

# SHAP sample limit for CPU efficiency
MAX_SHAP_SAMPLES = 1000


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ================================================================
# 1. LOAD DATA
# ================================================================

print_header("1. LOADING DATASET")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Failure count: {df[TARGET_COLUMN].sum()}")
print(f"Normal count: {(df[TARGET_COLUMN] == 0).sum()}")


# ================================================================
# 2. BUILD FEATURES
# ================================================================

print_header("2. BUILDING FEATURE MATRIX")

X = build_feature_frame(df)

if TARGET_COLUMN in X.columns:
    X = X.drop(columns=[TARGET_COLUMN])

y = df[TARGET_COLUMN].astype(int)

print("Features:")
for col in X.columns:
    print(f"  - {col}")


# ================================================================
# 3. EXACT FROZEN SPLIT
# ================================================================

print_header("3. RECREATING FROZEN SPLIT")

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.20,
    stratify=y_dev,
    random_state=RANDOM_STATE,
)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Final test samples : {len(X_test)}")


# ================================================================
# 4. BUILD AND TRAIN MODEL
# ================================================================

print_header("4. TRAINING OPTIMIZED XGBOOST")

preprocessor = build_preprocessor()

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
    **XGB_PARAMS,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model),
    ]
)

model.fit(X_train, y_train)

print("Optimized XGBoost trained successfully.")


# ================================================================
# 5. TRANSFORM TEST DATA
# ================================================================

print_header("5. TRANSFORMING TEST DATA")

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_xgb = model.named_steps["model"]

X_test_transformed = fitted_preprocessor.transform(X_test)

# Convert sparse matrix to dense if necessary
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

print(
    f"Transformed test shape: "
    f"{X_test_transformed.shape}"
)


# ================================================================
# 6. GET FEATURE NAMES
# ================================================================

print_header("6. EXTRACTING TRANSFORMED FEATURE NAMES")

feature_names = fitted_preprocessor.get_feature_names_out()

# Make names cleaner for presentation
clean_feature_names = []

for name in feature_names:

    name = name.replace(
        "num__",
        ""
    ).replace(
        "cat__",
        ""
    )

    clean_feature_names.append(name)

print("Transformed features:")

for name in clean_feature_names:
    print(f"  - {name}")


# ================================================================
# 7. SAMPLE TEST DATA FOR SHAP
# ================================================================

print_header("7. SELECTING SHAP SAMPLE")

rng = np.random.RandomState(RANDOM_STATE)

n_samples = min(
    MAX_SHAP_SAMPLES,
    len(X_test_transformed)
)

sample_indices = rng.choice(
    len(X_test_transformed),
    size=n_samples,
    replace=False,
)

X_shap = X_test_transformed[
    sample_indices
]

y_shap = y_test.iloc[
    sample_indices
].values

print(f"SHAP samples: {len(X_shap)}")
print(f"Failure samples: {y_shap.sum()}")


# ================================================================
# 8. CREATE SHAP EXPLAINER
# ================================================================

print_header("8. COMPUTING SHAP VALUES")

explainer = shap.TreeExplainer(
    fitted_xgb
)

shap_values = explainer.shap_values(
    X_shap
)

# For binary XGBoost this should normally be:
# samples x features
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(
    f"SHAP matrix shape: "
    f"{shap_values.shape}"
)


# ================================================================
# 9. CREATE SHAP DATAFRAME
# ================================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=clean_feature_names
)

X_shap_df = pd.DataFrame(
    X_shap,
    columns=clean_feature_names
)


# ================================================================
# 10. GLOBAL FEATURE IMPORTANCE
# ================================================================

print_header("10. GLOBAL SHAP FEATURE IMPORTANCE")

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": clean_feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap,
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
).reset_index(drop=True)

importance_df["Rank"] = (
    importance_df.index + 1
)

importance_path = (
    SHAP_DIR /
    "shap_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

print(
    importance_df.to_string(index=False)
)

print(
    f"\nSaved: {importance_path}"
)


# ================================================================
# 11. SHAP BAR PLOT
# ================================================================

print_header("11. GENERATING SHAP BAR PLOT")

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X_shap_df,
    plot_type="bar",
    show=False
)

plt.title(
    "Global SHAP Feature Importance - Optimized XGBoost"
)

plt.tight_layout()

bar_path = (
    FIGURE_DIR /
    "shap_global_bar.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {bar_path}")


# ================================================================
# 12. SHAP BEESWARM
# ================================================================

print_header("12. GENERATING SHAP BEESWARM")

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_shap_df,
    show=False
)

plt.title(
    "SHAP Summary Plot - Optimized XGBoost"
)

plt.tight_layout()

beeswarm_path = (
    FIGURE_DIR /
    "shap_beeswarm.png"
)

plt.savefig(
    beeswarm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {beeswarm_path}")


# ================================================================
# 13. SAVE ALL SHAP VALUES
# ================================================================

print_header("13. SAVING SHAP VALUES")

shap_values_output = X_shap_df.copy()

for feature in clean_feature_names:
    shap_values_output[
        f"SHAP_{feature}"
    ] = shap_df[feature]

shap_values_output["true_label"] = y_shap

# Model probability for the sampled rows
sample_probabilities = fitted_xgb.predict_proba(
    X_shap
)[:, 1]

shap_values_output[
    "failure_probability"
] = sample_probabilities

shap_values_path = (
    SHAP_DIR /
    "shap_values_samples.csv"
)

shap_values_output.to_csv(
    shap_values_path,
    index=False
)

print(
    f"Saved: {shap_values_path}"
)


# ================================================================
# 14. ANALYZE ACTUAL FAILURES
# ================================================================

print_header("14. FAILURE-SPECIFIC SHAP ANALYSIS")

failure_mask = (
    y_shap == 1
)

if failure_mask.sum() > 0:

    failure_shap = shap_values[
        failure_mask
    ]

    failure_importance = np.abs(
        failure_shap
    ).mean(axis=0)

    failure_importance_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Mean_Absolute_SHAP_Failure": (
            failure_importance
        ),
    })

    failure_importance_df = (
        failure_importance_df
        .sort_values(
            "Mean_Absolute_SHAP_Failure",
            ascending=False
        )
        .reset_index(drop=True)
    )

    failure_importance_df[
        "Rank"
    ] = failure_importance_df.index + 1

    failure_importance_path = (
        SHAP_DIR /
        "shap_failure_importance.csv"
    )

    failure_importance_df.to_csv(
        failure_importance_path,
        index=False
    )

    print(
        failure_importance_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {failure_importance_path}"
    )


# ================================================================
# 15. MISSED FAILURE SHAP ANALYSIS
# ================================================================

print_header("15. MISSED FAILURE SHAP ANALYSIS")

sample_predictions = (
    sample_probabilities >= 0.50
).astype(int)

missed_failure_mask = (
    (y_shap == 1) &
    (sample_predictions == 0)
)

print(
    f"Missed failures in SHAP sample: "
    f"{missed_failure_mask.sum()}"
)

if missed_failure_mask.sum() > 0:

    missed_shap = shap_values[
        missed_failure_mask
    ]

    missed_importance = np.abs(
        missed_shap
    ).mean(axis=0)

    missed_importance_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Mean_Absolute_SHAP_Missed": (
            missed_importance
        ),
    })

    missed_importance_df = (
        missed_importance_df
        .sort_values(
            "Mean_Absolute_SHAP_Missed",
            ascending=False
        )
        .reset_index(drop=True)
    )

    missed_importance_df[
        "Rank"
    ] = missed_importance_df.index + 1

    missed_path = (
        SHAP_DIR /
        "shap_missed_failure_importance.csv"
    )

    missed_importance_df.to_csv(
        missed_path,
        index=False
    )

    print(
        missed_importance_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {missed_path}"
    )


# ================================================================
# 16. RESCUED FAILURE SHAP ANALYSIS
# ================================================================

print_header("16. RESCUED FAILURE SHAP ANALYSIS")

# A failure is "rescued" if:
# probability < 0.50 but >= 0.20

rescued_mask = (
    (y_shap == 1) &
    (sample_probabilities < 0.50) &
    (sample_probabilities >= 0.20)
)

print(
    f"Rescued failures in SHAP sample: "
    f"{rescued_mask.sum()}"
)

if rescued_mask.sum() > 0:

    rescued_shap = shap_values[
        rescued_mask
    ]

    rescued_importance = np.abs(
        rescued_shap
    ).mean(axis=0)

    rescued_importance_df = pd.DataFrame({
        "Feature": clean_feature_names,
        "Mean_Absolute_SHAP_Rescued": (
            rescued_importance
        ),
    })

    rescued_importance_df = (
        rescued_importance_df
        .sort_values(
            "Mean_Absolute_SHAP_Rescued",
            ascending=False
        )
        .reset_index(drop=True)
    )

    rescued_importance_df[
        "Rank"
    ] = rescued_importance_df.index + 1

    rescued_path = (
        SHAP_DIR /
        "shap_rescued_failure_importance.csv"
    )

    rescued_importance_df.to_csv(
        rescued_path,
        index=False
    )

    print(
        rescued_importance_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {rescued_path}"
    )


# ================================================================
# 17. INDIVIDUAL MISSED FAILURE EXPLANATIONS
# ================================================================

print_header("17. INDIVIDUAL MISSED FAILURE EXPLANATIONS")

missed_indices = np.where(
    missed_failure_mask
)[0]

individual_rows = []

for local_idx in missed_indices:

    probability = sample_probabilities[
        local_idx
    ]

    row_shap = shap_values[
        local_idx
    ]

    top_indices = np.argsort(
        np.abs(row_shap)
    )[::-1][:5]

    for rank, feature_idx in enumerate(
        top_indices,
        start=1
    ):

        individual_rows.append({
            "sample_index": int(local_idx),
            "failure_probability": float(
                probability
            ),
            "feature": clean_feature_names[
                feature_idx
            ],
            "feature_value": float(
                X_shap[
                    local_idx,
                    feature_idx
                ]
            ),
            "shap_value": float(
                row_shap[feature_idx]
            ),
            "absolute_shap": float(
                abs(row_shap[feature_idx])
            ),
            "rank": rank,
        })


individual_df = pd.DataFrame(
    individual_rows
)

individual_path = (
    SHAP_DIR /
    "individual_missed_failure_explanations.csv"
)

individual_df.to_csv(
    individual_path,
    index=False
)

print(
    f"Saved: {individual_path}"
)


# ================================================================
# 18. SHAP DIRECTION ANALYSIS
# ================================================================

print_header("18. SHAP DIRECTION ANALYSIS")

direction_rows = []

for idx, feature in enumerate(
    clean_feature_names
):

    values = shap_values[:, idx]

    positive_count = (
        values > 0
    ).sum()

    negative_count = (
        values < 0
    ).sum()

    zero_count = (
        values == 0
    ).sum()

    direction_rows.append({
        "Feature": feature,
        "Positive_SHAP_Count": int(
            positive_count
        ),
        "Negative_SHAP_Count": int(
            negative_count
        ),
        "Zero_SHAP_Count": int(
            zero_count
        ),
        "Mean_SHAP": float(
            values.mean()
        ),
        "Mean_Absolute_SHAP": float(
            np.abs(values).mean()
        ),
    })


direction_df = pd.DataFrame(
    direction_rows
).sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

direction_path = (
    SHAP_DIR /
    "shap_direction_analysis.csv"
)

direction_df.to_csv(
    direction_path,
    index=False
)

print(
    direction_df.to_string(
        index=False
    )
)

print(
    f"\nSaved: {direction_path}"
)


# ================================================================
# 19. TOP FEATURE DEPENDENCE PLOTS
# ================================================================

print_header("19. GENERATING TOP FEATURE DEPENDENCE PLOTS")

top_features = importance_df[
    "Feature"
].head(3).tolist()

for feature in top_features:

    feature_index = clean_feature_names.index(
        feature
    )

    plt.figure(figsize=(9, 6))

    shap.dependence_plot(
        feature_index,
        shap_values,
        X_shap_df,
        show=False
    )

    plt.title(
        f"SHAP Dependence - {feature}"
    )

    plt.tight_layout()

    safe_feature = (
        feature
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "_")
        .replace("/", "_")
    )

    path = (
        FIGURE_DIR /
        f"shap_dependence_{safe_feature}.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ================================================================
# 20. FINAL SUMMARY
# ================================================================

print_header("20. FINAL SHAP SUMMARY")

print("\nTop global features:")

print(
    importance_df[
        [
            "Rank",
            "Feature",
            "Mean_Absolute_SHAP"
        ]
    ].head(10).to_string(
        index=False
    )
)

print("\nOutput directory:")
print(SHAP_DIR)

print("\nStage 18 SHAP analysis completed successfully.")
