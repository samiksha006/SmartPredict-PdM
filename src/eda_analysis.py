"""
SmartPredict-PdM
Stage 10A: Final EDA Analysis

Generates publication-ready EDA figures for the
Advanced Machine Learning Assignment II.
"""

from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import DATA_PATH, RESULTS_FIGURE_DIR

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SMARTPREDICT-PdM")
print("FINAL EDA ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape: {df.shape}")

print("\nColumns:")
print(list(df.columns))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# TARGET INFORMATION
# ============================================================

target = "Machine failure"

failure_counts = df[target].value_counts().sort_index()

normal_count = int(failure_counts.get(0, 0))
failure_count = int(failure_counts.get(1, 0))

failure_rate = failure_count / len(df) * 100

print("\n" + "=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)

print(f"Normal machines : {normal_count}")
print(f"Failures        : {failure_count}")
print(f"Failure rate    : {failure_rate:.2f}%")


# ============================================================
# FIGURE 1 — CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

labels = ["Normal", "Machine Failure"]
values = [normal_count, failure_count]

plt.bar(labels, values)

plt.title(
    "Machine Failure Class Distribution",
    fontsize=15,
    fontweight="bold"
)

plt.ylabel("Number of Samples")

for i, value in enumerate(values):
    percentage = value / len(df) * 100

    plt.text(
        i,
        value + max(values) * 0.015,
        f"{value:,}\n({percentage:.2f}%)",
        ha="center",
        fontsize=11
    )

plt.tight_layout()

plt.savefig(
    RESULTS_FIGURE_DIR / "eda_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# NUMERIC FEATURES
# ============================================================

numeric_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ============================================================
# FIGURE 2 — FEATURE DISTRIBUTIONS
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 12)
)

axes = axes.flatten()

for i, feature in enumerate(numeric_features):

    axes[i].hist(
        df[feature],
        bins=30
    )

    axes[i].set_title(
        feature,
        fontsize=12,
        fontweight="bold"
    )

    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Frequency")

# Hide unused subplot
axes[-1].axis("off")

fig.suptitle(
    "Distribution of Machine Operating Features",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.savefig(
    RESULTS_FIGURE_DIR / "eda_feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 3 — CORRELATION HEATMAP
# ============================================================

correlation_features = numeric_features + [
    target
]

corr = df[
    correlation_features
].corr()

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    corr,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title(
    "Correlation Matrix of Machine Variables",
    fontsize=15,
    fontweight="bold"
)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):

        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )

plt.tight_layout()

plt.savefig(
    RESULTS_FIGURE_DIR / "eda_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 4 — NORMAL VS FAILURE FEATURE COMPARISON
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(12, 12)
)

axes = axes.flatten()

for i, feature in enumerate(numeric_features):

    normal_values = df.loc[
        df[target] == 0,
        feature
    ]

    failure_values = df.loc[
        df[target] == 1,
        feature
    ]

    axes[i].boxplot(
        [
            normal_values,
            failure_values
        ],
        labels=[
            "Normal",
            "Failure"
        ]
    )

    axes[i].set_title(
        feature,
        fontsize=12,
        fontweight="bold"
    )

    axes[i].set_ylabel(
        "Value"
    )

axes[-1].axis("off")

fig.suptitle(
    "Operating Conditions: Normal vs Machine Failure",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.96]
)

plt.savefig(
    RESULTS_FIGURE_DIR /
    "eda_normal_vs_failure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 5 — TORQUE VS ROTATIONAL SPEED
# ============================================================

plt.figure(
    figsize=(10, 7)
)

normal = df[target] == 0
failure = df[target] == 1

plt.scatter(
    df.loc[normal, "Rotational speed [rpm]"],
    df.loc[normal, "Torque [Nm]"],
    alpha=0.35,
    s=15,
    label="Normal"
)

plt.scatter(
    df.loc[failure, "Rotational speed [rpm]"],
    df.loc[failure, "Torque [Nm]"],
    alpha=0.8,
    s=30,
    label="Failure"
)

plt.xlabel(
    "Rotational Speed [rpm]"
)

plt.ylabel(
    "Torque [Nm]"
)

plt.title(
    "Torque vs Rotational Speed by Machine Failure",
    fontsize=15,
    fontweight="bold"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS_FIGURE_DIR /
    "eda_torque_vs_speed.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 6 — TOOL WEAR VS FAILURE
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.boxplot(
    [
        df.loc[
            df[target] == 0,
            "Tool wear [min]"
        ],
        df.loc[
            df[target] == 1,
            "Tool wear [min]"
        ]
    ],
    labels=[
        "Normal",
        "Failure"
    ]
)

plt.xlabel(
    "Machine Status"
)

plt.ylabel(
    "Tool Wear [min]"
)

plt.title(
    "Tool Wear Distribution by Machine Failure",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    RESULTS_FIGURE_DIR /
    "eda_tool_wear_failure.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 7 — FAILURE RATE BY MACHINE TYPE
# ============================================================

type_failure = (
    df.groupby("Type")[target]
    .mean()
    .sort_index()
    * 100
)

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    type_failure.index,
    type_failure.values
)

plt.xlabel(
    "Machine Type"
)

plt.ylabel(
    "Failure Rate (%)"
)

plt.title(
    "Machine Failure Rate by Product Type",
    fontsize=15,
    fontweight="bold"
)

for i, value in enumerate(type_failure.values):

    plt.text(
        i,
        value + 0.05,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    RESULTS_FIGURE_DIR /
    "eda_failure_by_type.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EDA FIGURES GENERATED")
print("=" * 70)

figures = [
    "eda_class_distribution.png",
    "eda_feature_distributions.png",
    "eda_correlation_heatmap.png",
    "eda_normal_vs_failure.png",
    "eda_torque_vs_speed.png",
    "eda_tool_wear_failure.png",
    "eda_failure_by_type.png"
]

for figure in figures:

    path = RESULTS_FIGURE_DIR / figure

    print(path)


print("\n" + "=" * 70)
print("EDA COMPLETED")
print("=" * 70)