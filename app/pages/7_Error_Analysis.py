import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app_utils import configure_page, section_title, TABLES_DIR, FINAL_ERROR_DIR, render_footer

configure_page("SmartPredict-PdM | Error Analysis")

st.html("""
<div class="sp-hero">
  <div style="font-size:12px;font-weight:750;color:#dc2626;letter-spacing:.10em;text-transform:uppercase;">FAILURE ANALYSIS</div>
  <div class="sp-hero-title" style="font-size:34px;">Error Analysis</div>
  <div class="sp-hero-subtitle">Inspect missed failures, false alarms and threshold-dependent error patterns.</div>
</div>
""")

summary=pd.read_csv(FINAL_ERROR_DIR/"threshold_error_summary.csv")
records=pd.read_csv(FINAL_ERROR_DIR/"final_error_records.csv")

outputs=st.multiselect(
    "Select outputs to view",
    ["Error summary","Confusion matrix","Missed failures","False positives","Probability distribution","Feature means"],
    default=["Error summary"],
)

if "Error summary" in outputs:
    section_title("Error summary","Choose an operating point from the frozen final-test analysis.")
    point=st.selectbox("Operating point",summary["Operating Point"].tolist(),key="err_point")
    r=summary[summary["Operating Point"].eq(point)].iloc[0]
    c=st.columns(5)
    c[0].metric("Threshold",f"{r.Threshold:.2f}")
    c[1].metric("Recall",f"{r.Recall*100:.2f}%")
    c[2].metric("Precision",f"{r.Precision*100:.2f}%")
    c[3].metric("False positives",f"{int(r.FP)}")
    c[4].metric("False negatives",f"{int(r.FN)}")

if "Confusion matrix" in outputs:
    section_title("Confusion matrix","Select the same operating point to inspect counts.")
    point=st.selectbox("Confusion-matrix point",summary["Operating Point"].tolist(),key="err_cm_point")
    r=summary[summary["Operating Point"].eq(point)].iloc[0]
    cm=pd.DataFrame([[int(r.TN),int(r.FP)],[int(r.FN),int(r.TP)]],index=["Actual normal","Actual failure"],columns=["Predicted normal","Predicted failure"])
    st.dataframe(cm,use_container_width=True)

if "Missed failures" in outputs:
    section_title("Missed failures","Records classified as normal when a failure occurred.")
    threshold_file="missed_failures_default_0.50.csv"
    d=pd.read_csv(FINAL_ERROR_DIR/threshold_file)
    n=st.slider("Rows",5,min(50,max(5,len(d))),min(10,max(5,len(d))),key="fn_rows")
    st.dataframe(d.head(n),use_container_width=True,hide_index=True)

if "False positives" in outputs:
    section_title("False positives","False alarms under the default operating point.")
    d=pd.read_csv(FINAL_ERROR_DIR/"false_positives_default_0.50.csv")
    n=st.slider("Rows",5,min(50,max(5,len(d))),min(10,max(5,len(d))),key="fp_rows")
    st.dataframe(d.head(n),use_container_width=True,hide_index=True)

if "Probability distribution" in outputs:
    section_title("Probability distribution","Model scores on the final test set.")
    d=pd.read_csv(FINAL_ERROR_DIR/"final_test_probabilities.csv")
    prob_col=next((c for c in d.columns if "prob" in c.lower()),d.columns[-1])
    fig,ax=plt.subplots(figsize=(9,4.5))
    ax.hist(d[prob_col],bins=30)
    ax.set_xlabel("Predicted failure probability"); ax.set_ylabel("Count"); ax.set_title("Final-test probability distribution")
    ax.grid(axis="y",alpha=.15); ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "Feature means" in outputs:
    section_title("Feature means","Observed averages by error category. These are associations, not causal effects.")
    p=FINAL_ERROR_DIR/"default_threshold_feature_means.csv"
    if p.exists():
        st.dataframe(pd.read_csv(p),use_container_width=True,hide_index=True)
    else:
        st.info("Feature-mean output is not available in the supplied results folder.")

render_footer()
