from pathlib import Path
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

DATA_PATH = DATA_DIR / "ai4i2020.csv"

FINAL_ERROR_DIR = RESULTS_DIR / "final_error_analysis"
FINAL_SHAP_DIR = RESULTS_DIR / "final_shap_analysis"

TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = [
    "Type"
]


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = {
    "Default reference — 0.50": 0.50,
    "Enhanced-model validation — 0.16": 0.16,
    "Lower threshold — 0.11": 0.11,
    "High sensitivity reference — 0.05": 0.05,
}


# ============================================================
# GLOBAL PAGE CONFIGURATION
# ============================================================

def configure_page(title="SmartPredict-PdM"):
    """
    Configure the shared SmartPredict-PdM application UI.

    This function is called from every Streamlit page.
    It controls only page configuration and styling.
    """

    st.set_page_config(
        page_title=title,
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    css = """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background: #ffffff;
    }

    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }

    [data-testid="stMain"] {
        background: #ffffff;
    }

    [data-testid="stHeader"] {
        background: #ffffff;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1,
    h2,
    h3 {
        color: #0f172a !important;
    }

    h1 {
        font-weight: 750 !important;
        letter-spacing: -0.025em;
    }

    h2 {
        font-weight: 700 !important;
    }

    h3 {
        font-weight: 650 !important;
    }

    p {
        color: #475569;
        line-height: 1.65;
    }

    .stCaption {
        color: #64748b !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] > div:first-child {
        background: #f8fafc;
    }

    [data-testid="stSidebar"] p {
        color: #475569;
    }

    [data-testid="stSidebar"] .stRadio > div {
        gap: 4px;
    }

    [data-testid="stSidebar"] .stRadio label {
        border-radius: 9px;
        padding: 8px 10px;
        transition: background 0.15s ease;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background: #eef2ff;
    }


    /* ========================================================
       SIDEBAR BRAND
       ======================================================== */

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 4px 4px 14px 4px;
    }

    .sidebar-logo {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: #2563eb;
        color: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 19px;
        font-weight: 750;
        flex-shrink: 0;
    }

    .sidebar-brand-name {
        font-size: 17px;
        font-weight: 750;
        color: #0f172a;
        line-height: 1.2;
    }

    .sidebar-brand-subtitle {
        font-size: 10px;
        color: #64748b;
        margin-top: 2px;
    }

    .sidebar-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 18px 0;
    }

    .sidebar-section-label {
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.10em;
        color: #94a3b8;
        text-transform: uppercase;
        margin: 16px 4px 7px 4px;
    }

    .model-status {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 11px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        border-radius: 9px;
        margin-top: 10px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #16a34a;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .status-title {
        font-size: 11px;
        font-weight: 700;
        color: #334155;
    }

    .status-subtitle {
        font-size: 10px;
        color: #94a3b8;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    .sp-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }

    .sp-card-title {
        font-size: 14px;
        font-weight: 650;
        color: #475569;
        margin-bottom: 7px;
    }

    .sp-card-value {
        font-size: 30px;
        line-height: 1.1;
        font-weight: 750;
        color: #0f172a;
    }

    .sp-card-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-top: 7px;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .sp-hero {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #ffffff 100%
        );

        border: 1px solid #e2e8f0;
        border-radius: 18px;

        padding: 32px;
        margin-bottom: 25px;

        box-shadow:
            0 4px 18px rgba(15, 23, 42, 0.04);
    }

    .sp-hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
    }

    .sp-hero-subtitle {
        margin-top: 8px;
        font-size: 17px;
        color: #64748b;
    }


    /* ========================================================
       STATUS BOXES
       ======================================================== */

    .status-normal {
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #047857;

        padding: 17px 20px;
        border-radius: 12px;

        font-size: 16px;
        font-weight: 700;

        margin: 15px 0;
    }

    .status-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #b45309;

        padding: 17px 20px;
        border-radius: 12px;

        font-size: 16px;
        font-weight: 700;

        margin: 15px 0;
    }

    .status-danger {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #b91c1c;

        padding: 17px 20px;
        border-radius: 12px;

        font-size: 16px;
        font-weight: 700;

        margin: 15px 0;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 9px;
        min-height: 44px;

        font-weight: 650;

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background: #ffffff;

        border: 1px solid #e2e8f0;
        border-radius: 12px;

        padding: 16px 18px;

        min-height: 100px;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 750 !important;
    }


    /* ========================================================
       INPUTS
       ======================================================== */

    .stTextInput input,
    .stNumberInput input {
        border-radius: 8px !important;
    }

    div[data-baseweb="select"] > div {
        border-radius: 8px;
    }


    /* ========================================================
       DATAFRAMES
       ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        background: #ffffff;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px;
    }


    /* ========================================================
       SIDEBAR STATUS
       ======================================================== */

    .model-status {
        display: flex;
        align-items: center;
        gap: 9px;

        padding: 11px;

        background: #ffffff;

        border: 1px solid #e2e8f0;
        border-radius: 9px;
    }

    .status-dot {
        width: 7px;
        height: 7px;

        background: #16a34a;

        border-radius: 50%;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        margin-top: 40px;

        padding-top: 18px;

        border-top: 1px solid #e2e8f0;

        color: #94a3b8;

        font-size: 11px;

        text-align: center;
    }


    /* ========================================================
       STREAMLIT CHROME
       ======================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ========================================================
       RESPONSIVE
       ======================================================== */

    @media (max-width: 900px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .sp-hero {
            padding: 22px;
        }

        .sp-hero-title {
            font-size: 30px;
        }
    }

    </style>
    """

    st.markdown(
        css,
        unsafe_allow_html=True,
    )

# ============================================================
# SIDEBAR BRAND
# ============================================================

def render_sidebar_brand():
    """
    Shared SmartPredict-PdM sidebar branding.
    """

    st.sidebar.html(
        """
        <div class="sidebar-brand">

            <div class="sidebar-logo">
                ⚙
            </div>

            <div>

                <div class="sidebar-brand-name">
                    SmartPredict-PdM
                </div>

                <div class="sidebar-brand-subtitle">
                    Industrial Predictive Intelligence
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# FIND MODEL
# ============================================================

def find_model_path():

    possible_names = [
        "final_experiment/advanced_features_scheduled_xgb.joblib",
        "advanced_features_scheduled_xgb.joblib",
        "xgboost_optimized.joblib",
        "optimized_xgboost.joblib",
        "xgboost_optimized.pkl",
        "optimized_xgboost.pkl",
        "xgb_optimized.joblib",
        "xgb_optimized.pkl",
        "final_xgboost.joblib",
        "final_xgboost.pkl",
    ]

    for filename in possible_names:

        path = MODELS_DIR / filename

        if path.exists():
            return path


    if MODELS_DIR.exists():

        for path in MODELS_DIR.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in [
                ".joblib",
                ".pkl",
            ]:
                continue

            filename = path.name.lower()

            if (
                "xgb" in filename
                or "xgboost" in filename
                or "optimized" in filename
            ):
                return path


    return None


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = find_model_path()

    if model_path is None:

        raise FileNotFoundError(
            "Optimized XGBoost model could not be found.\n\n"
            f"Model directory:\n{MODELS_DIR}"
        )

    return joblib.load(model_path)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

def create_input_dataframe(
    machine_type,
    air_temperature,
    process_temperature,
    rotational_speed,
    torque,
    tool_wear,
):

    return pd.DataFrame(
        [
            {
                "Type": machine_type,

                "Air temperature [K]":
                    air_temperature,

                "Process temperature [K]":
                    process_temperature,

                "Rotational speed [rpm]":
                    rotational_speed,

                "Torque [Nm]":
                    torque,

                "Tool wear [min]":
                    tool_wear,
            }
        ],
        columns=FEATURES,
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_failure(
    model,
    input_df,
    threshold=0.50,
):

    probabilities = model.predict_proba(
        input_df
    )

    failure_probability = float(
        probabilities[0, 1]
    )

    prediction = int(
        failure_probability >= threshold
    )

    return (
        prediction,
        failure_probability,
    )


# ============================================================
# SHAP EXPLANATION
# ============================================================

def get_shap_explanation(
    model,
    input_df,
):

    if not hasattr(
        model,
        "named_steps",
    ):

        raise ValueError(
            "Expected an sklearn Pipeline."
        )


    preprocessor = None
    xgb_model = None


    for _, step in model.named_steps.items():

        if (
            hasattr(step, "transform")
            and hasattr(
                step,
                "get_feature_names_out",
            )
        ):

            preprocessor = step


        if hasattr(
            step,
            "get_booster",
        ):

            xgb_model = step


    if (
        preprocessor is None
        or xgb_model is None
    ):

        raise ValueError(
            "Could not identify preprocessing "
            "and XGBoost steps."
        )


    model_input = input_df
    feature_engineering = model.named_steps.get("feature_engineering")
    if feature_engineering is not None:
        model_input = feature_engineering.transform(input_df)

    transformed = preprocessor.transform(
        model_input
    )


    if hasattr(
        transformed,
        "toarray",
    ):

        transformed = transformed.toarray()


    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    clean_feature_names = [

        str(name)
        .replace(
            "num__",
            "",
        )
        .replace(
            "cat__",
            "",
        )

        for name in feature_names
    ]


    explainer = shap.TreeExplainer(
        xgb_model
    )


    shap_values = explainer.shap_values(
        transformed
    )


    if isinstance(
        shap_values,
        list,
    ):

        shap_values = shap_values[1]


    shap_values = np.asarray(
        shap_values
    )


    if shap_values.ndim == 2:

        shap_values = shap_values[0]


    return (
        transformed,
        clean_feature_names,
        shap_values,
    )


# ============================================================
# METRIC CARD
# ============================================================

def metric_card(
    title,
    value,
    subtitle="",
):

    html = dedent(
        f"""
        <div class="sp-card">

            <div class="sp-card-title">
                {title}
            </div>

            <div class="sp-card-value">
                {value}
            </div>

            <div class="sp-card-subtitle">
                {subtitle}
            </div>

        </div>
        """
    )

    st.html(html)


# ============================================================
# SECTION TITLE
# ============================================================

def section_title(
    title,
    description=None,
):

    st.markdown(
        f"## {title}"
    )

    if description:

        st.caption(
            description
        )


# ============================================================
# RESULT CSV
# ============================================================

@st.cache_data
def load_result_csv(path):

    path = Path(path)

    if not path.exists():
        return None

    try:

        return pd.read_csv(path)

    except Exception:

        return None


# ============================================================
# SHAP PATHS
# ============================================================

def get_shap_paths():

    return {

        "global":
            FINAL_SHAP_DIR
            / "shap_feature_importance.csv",

        "failure":
            FINAL_SHAP_DIR
            / "shap_failure_importance.csv",

        "missed":
            FINAL_SHAP_DIR
            / "shap_missed_failure_importance.csv",

        "rescued":
            FINAL_SHAP_DIR
            / "shap_rescued_failure_importance.csv",

        "direction":
            FINAL_SHAP_DIR
            / "shap_direction_analysis.csv",

        "individual":
            FINAL_SHAP_DIR
            / "individual_missed_failure_explanations.csv",

        "global_plot":
            FINAL_SHAP_DIR
            / "figures"
            / "shap_global_bar.png",

        "beeswarm":
            FINAL_SHAP_DIR
            / "figures"
            / "shap_beeswarm.png",
    }


# ============================================================
# ERROR ANALYSIS PATHS
# ============================================================

def get_error_analysis_paths():

    return {

        "summary":
            FINAL_ERROR_DIR
            / "threshold_error_summary.csv",

        "records":
            FINAL_ERROR_DIR
            / "final_error_records.csv",

        "probabilities":
            FINAL_ERROR_DIR
            / "final_test_probabilities.csv",

        "feature_means":
            FINAL_ERROR_DIR
            / "default_threshold_feature_means.csv",

        "missed_failures":
            FINAL_ERROR_DIR
            / "missed_failures_default_0.50.csv",

        "detected_failures":
            FINAL_ERROR_DIR
            / "detected_failures_default_0.50.csv",

        "probability_summary":
            FINAL_ERROR_DIR
            / "detected_vs_missed_probability_summary.csv",
    }


# ============================================================
# FOOTER
# ============================================================

def render_footer():

    st.html(
        """
        <div class="app-footer">
            SmartPredict-PdM · Cost-Aware Industrial
            Predictive Maintenance
        </div>
        """
    )
