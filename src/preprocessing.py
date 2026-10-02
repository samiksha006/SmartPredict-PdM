import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ID_COLUMNS, LEAKAGE_PRONE_COLUMNS

def build_feature_frame(df):
    drop_cols = [c for c in ID_COLUMNS + LEAKAGE_PRONE_COLUMNS if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore").copy()

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])

# ============================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================
ADVANCED_NUMERIC_FEATURES = NUMERIC_FEATURES + [
    "Temperature Difference [K]",
    "Temperature Ratio",
    "Mechanical Power Proxy",
    "Torque per Speed",
    "Wear per Speed",
    "Thermal-Wear Interaction",
]


def add_advanced_features(df):
    from model_utils import add_advanced_features as _add
    return _add(df)


def build_advanced_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, ADVANCED_NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
