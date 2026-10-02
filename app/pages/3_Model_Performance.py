import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app_utils import configure_page, section_title, TABLES_DIR, FIGURES_DIR, render_footer

configure_page("SmartPredict-PdM | Model Performance")

st.html("""
<div class="sp-hero">
  <div style="font-size:12px;font-weight:750;color:#2563eb;letter-spacing:.10em;text-transform:uppercase;">MODEL EVALUATION</div>
  <div class="sp-hero-title" style="font-size:34px;">Model Performance</div>
  <div class="sp-hero-subtitle">Compare kernel machines, ensembles and the primary XGBoost model using selectable evidence.</div>
</div>
""")

master = pd.read_csv(TABLES_DIR/"final_experiment"/"master_test_results.csv")
ensemble = pd.read_csv(TABLES_DIR/"clean_ensemble"/"clean_ensemble_results.csv")
cv = pd.read_csv(TABLES_DIR/"repeated_cv"/"repeated_cv_summary.csv")

outputs = st.multiselect(
    "Select outputs to view",
    ["Primary model metrics","Model comparison table","Comparison chart","Repeated-CV results","Confusion matrix","Metric definitions"],
    default=["Primary model metrics"],
)

if "Primary model metrics" in outputs:
    section_title("Primary model", "Optimized XGBoost frozen-test reference.")
    row=master[master["Model"].eq("XGBoost")].iloc[0]
    c=st.columns(6)
    for col,label,key in zip(c,["Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC"],["Accuracy","Precision","Recall","F1","ROC_AUC","PR_AUC"]):
        col.metric(label,f"{row[key]*100:.2f}%")
    st.caption(f"MCC {row['MCC']:.4f} · threshold {row['Threshold']:.2f} · TP {int(row['TP'])} · FP {int(row['FP'])} · FN {int(row['FN'])} · TN {int(row['TN'])}")

if "Model comparison table" in outputs:
    section_title("Model comparison", "Final frozen experiment across individual and ensemble models.")
    metrics=["Accuracy","Precision","Recall","F1","ROC_AUC","PR_AUC","MCC"]
    chosen=st.multiselect("Metrics to display",metrics,default=["F1","PR_AUC","MCC"],key="model_table_metrics")
    st.dataframe(master[["Model","Threshold"]+chosen].round(4),use_container_width=True,hide_index=True)

if "Comparison chart" in outputs:
    section_title("Comparison chart", "Select the metric and model family you want to visualize.")
    metric=st.selectbox("Metric",["F1","PR_AUC","Recall","MCC","Precision","ROC_AUC"],key="model_chart_metric")
    family=st.radio("Model set",["All final models","Ensemble-focused"],horizontal=True,key="model_chart_family")
    d=master.copy()
    if family=="Ensemble-focused":
        d=d[d["Model"].isin(["Random Forest","Extra Trees","AdaBoost","XGBoost"])]
    d=d.sort_values(metric,ascending=True)
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(d["Model"],d[metric]); ax.set_xlabel(metric); ax.set_title(f"{metric} comparison")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",alpha=.15)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "Repeated-CV results" in outputs:
    section_title("Repeated stratified CV", "5-fold × 3-repeat development-set summary.")
    chosen=st.multiselect("CV metrics",["PR_AUC_Mean","Recall_Mean","F1_Mean","MCC_Mean","ROC_AUC_Mean"],default=["PR_AUC_Mean","F1_Mean"],key="cv_metrics")
    st.dataframe(cv[["Stage"]+chosen].round(4),use_container_width=True,hide_index=True)

if "Confusion matrix" in outputs:
    section_title("Confusion matrix", "Select a model and inspect its frozen-test counts.")
    model=st.selectbox("Model",master["Model"].tolist(),key="cm_model")
    r=master[master["Model"].eq(model)].iloc[0]
    cm=pd.DataFrame([[int(r.TN),int(r.FP)],[int(r.FN),int(r.TP)]],index=["Actual normal","Actual failure"],columns=["Predicted normal","Predicted failure"])
    st.dataframe(cm,use_container_width=True)
    st.caption(f"Threshold: {r['Threshold']:.2f}")

if "Metric definitions" in outputs:
    section_title("Metric definitions", "Why multiple metrics are used.")
    st.markdown("""
- **Recall:** proportion of actual failures detected.
- **Precision:** proportion of predicted failures that are actual failures.
- **F1:** harmonic mean of precision and recall.
- **PR-AUC:** useful for ranking performance under class imbalance.
- **MCC:** balanced correlation-style classification measure.
- **ROC-AUC:** threshold-independent ranking measure.
""")

render_footer()
