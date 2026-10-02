from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATA
# ============================================================

DATA_DIR = BASE_DIR / "data"

DATA_PATH = DATA_DIR / "ai4i2020.csv"

TARGET_COLUMN = "Machine failure"
TARGET = TARGET_COLUMN
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# Identifier columns
ID_COLUMNS = [
    "UDI",
    "Product ID"
]


# Categorical features
CATEGORICAL_FEATURES = [
    "Type"
]


# Numeric features used for prediction
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# Failure-mode columns are excluded because
# they are highly related to the target and can cause
# target leakage in a pre-failure prediction setting.
LEAKAGE_PRONE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]


# ============================================================
# RESULTS
# ============================================================

RESULTS_DIR = BASE_DIR / "results"

RESULTS_TABLE_DIR = RESULTS_DIR / "tables"

RESULTS_FIGURE_DIR = RESULTS_DIR / "figures"


# ============================================================
# MODELS
# ============================================================

MODELS_DIR = BASE_DIR / "models"
MODEL_DIR = MODELS_DIR
TABLES_DIR = RESULTS_TABLE_DIR


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)