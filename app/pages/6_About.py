import streamlit as st
from app_utils import configure_page, section_title, render_footer

configure_page("SmartPredict-PdM | About")

st.html("""
<div class="sp-hero">
  <div style="font-size:12px;font-weight:750;color:#2563eb;letter-spacing:.10em;text-transform:uppercase;">SMART MANUFACTURING · INDUSTRY 4.0</div>
  <div class="sp-hero-title" style="font-size:36px;">SmartPredict-PdM</div>
  <div class="sp-hero-subtitle">An experimental framework for cost-aware industrial predictive maintenance.</div>
</div>
""")

sections=st.multiselect(
    "Select information to view",
    ["Project objective","Dataset","Predictive features","Methodology","Evaluation protocol","Primary model","Final test results","Technology stack","Research focus"],
    default=["Project objective"],
)

if "Project objective" in sections:
    section_title("Project objective","Research focus of the application.")
    st.markdown("""
**Research objective**

Experimentally investigate whether optimized kernel machines and ensemble learning techniques can improve detection of rare industrial machine failures while reducing false-negative predictions.

**Research questions**
- How do different SVM kernels affect machine-failure prediction?
- How do ensemble models compare with individual kernel-based models?
- What is the effect of hyperparameter optimization?
- Which evaluation metrics are informative under class imbalance?
- How do operating thresholds change false-negative risk?
""")

if "Dataset" in sections:
    section_title("Dataset","AI4I 2020 Predictive Maintenance Dataset.")
    st.markdown("10,000 observations and 14 columns. The target contains 339 failures (3.39%). Model inputs exclude identifiers and leakage-prone failure-mode indicators.")

if "Predictive features" in sections:
    section_title("Predictive features","Features used for pre-failure modelling.")
    st.dataframe({"Feature":["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]},use_container_width=True,hide_index=True)

if "Methodology" in sections:
    section_title("Methodology","Experimental workflow.")
    st.markdown("EDA → preprocessing → baseline models → kernel machines → ensemble models → optimization → imbalance analysis → repeated CV → cost-aware thresholds → frozen final test → error analysis → SHAP.")

if "Evaluation protocol" in sections:
    section_title("Evaluation protocol","Frozen split used by the final analysis.")
    st.markdown("""
- Training: **6,400**
- Validation: **1,600**
- Final test: **2,000**
- Thresholds selected on validation.
- Final test kept untouched until final evaluation.
""")

if "Primary model" in sections:
    section_title("Primary model","Optimized XGBoost is the primary predictive model in the application.")
    st.markdown("The project also evaluates logistic regression, decision tree, KNN, linear/polynomial/RBF SVM, Random Forest, Extra Trees and AdaBoost.")

if "Final test results" in sections:
    section_title("Final test results","Default 0.50 operating point.")
    c=st.columns(6)
    vals=[("Accuracy","98.65%"),("Precision","90.20%"),("Recall","67.65%"),("F1","77.31%"),("ROC-AUC","97.34%"),("PR-AUC","82.04%")]
    for col,(a,b) in zip(c,vals): col.metric(a,b)

if "Technology stack" in sections:
    section_title("Technology stack","Implementation components.")
    st.markdown("Python · pandas · NumPy · scikit-learn · XGBoost · SHAP · Streamlit · Matplotlib")

if "Research focus" in sections:
    section_title("Research focus","What the application emphasizes.")
    st.markdown("Kernel machines, ensemble learning, model evaluation, hyperparameter optimization, class imbalance, false-negative sensitivity, threshold selection and explainability.")

render_footer()
