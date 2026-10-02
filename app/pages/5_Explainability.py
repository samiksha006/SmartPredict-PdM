import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app_utils import configure_page, section_title, FINAL_SHAP_DIR, render_footer

configure_page("SmartPredict-PdM | Explainability")

st.html("""
<div class="sp-hero">
  <div style="font-size:12px;font-weight:750;color:#7c3aed;letter-spacing:.10em;text-transform:uppercase;">MODEL INTERPRETABILITY</div>
  <div class="sp-hero-title" style="font-size:34px;">Explainability</div>
  <div class="sp-hero-subtitle">Inspect SHAP evidence only when you need it.</div>
</div>
""")

outputs=st.multiselect(
    "Select explainability outputs",
    ["SHAP summary","Global feature importance","Failure-specific importance","Missed-failure importance","Rescued-failure importance","Direction analysis","Dependence plot","Individual missed-failure explanations"],
    default=["SHAP summary"],
)

if "SHAP summary" in outputs:
    section_title("SHAP summary","Global model-interpretation overview.")
    c=st.columns(3)
    c[0].metric("SHAP samples","1,000")
    c[1].metric("Input features","8 transformed")
    c[2].metric("Primary model","Optimized XGBoost")
    st.caption("SHAP indicates model attribution; it should not be interpreted as causal evidence.")

if "Global feature importance" in outputs:
    section_title("Global feature importance","Mean absolute SHAP values.")
    d=pd.read_csv(FINAL_SHAP_DIR/"shap_feature_importance.csv").sort_values("Mean_Absolute_SHAP")
    fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(d["Feature"],d["Mean_Absolute_SHAP"])
    ax.set_xlabel("Mean |SHAP|"); ax.set_title("Global feature importance")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",alpha=.15)
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.dataframe(d.sort_values("Rank"),use_container_width=True,hide_index=True)

if "Failure-specific importance" in outputs:
    section_title("Failure-specific importance","Attribution calculated for failure samples.")
    d=pd.read_csv(FINAL_SHAP_DIR/"shap_failure_importance.csv")
    st.dataframe(d,use_container_width=True,hide_index=True)

if "Missed-failure importance" in outputs:
    section_title("Missed-failure importance","Attribution for sampled missed failures.")
    d=pd.read_csv(FINAL_SHAP_DIR/"shap_missed_failure_importance.csv")
    st.dataframe(d,use_container_width=True,hide_index=True)

if "Rescued-failure importance" in outputs:
    section_title("Rescued-failure importance","Attribution for sampled rescued failures.")
    d=pd.read_csv(FINAL_SHAP_DIR/"shap_rescued_failure_importance.csv")
    st.dataframe(d,use_container_width=True,hide_index=True)

if "Direction analysis" in outputs:
    section_title("Direction analysis","Positive/negative SHAP contribution counts and mean values.")
    d=pd.read_csv(FINAL_SHAP_DIR/"shap_direction_analysis.csv")
    st.dataframe(d,use_container_width=True,hide_index=True)

if "Dependence plot" in outputs:
    section_title("Dependence plot","Choose one feature relationship.")
    options={
        "Torque [Nm]":FINAL_SHAP_DIR/"figures"/"shap_dependence_Torque_Nm.png",
        "Tool wear [min]":FINAL_SHAP_DIR/"figures"/"shap_dependence_Tool_wear_min.png",
        "Rotational speed [rpm]":FINAL_SHAP_DIR/"figures"/"shap_dependence_Rotational_speed_rpm.png",
    }
    f=st.selectbox("Feature",list(options),key="shap_depend")
    st.image(str(options[f]),use_container_width=True)

if "Individual missed-failure explanations" in outputs:
    section_title("Individual missed-failure explanations","Select a record from the supplied explanation table.")
    d=pd.read_csv(FINAL_SHAP_DIR/"individual_missed_failure_explanations.csv")
    st.dataframe(d,use_container_width=True,hide_index=True)

render_footer()
