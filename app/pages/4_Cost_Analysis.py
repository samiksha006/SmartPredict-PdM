import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from app_utils import configure_page, section_title, TABLES_DIR, FIGURES_DIR, render_footer

configure_page("SmartPredict-PdM | Cost Analysis")

st.html("""
<div class="sp-hero">
  <div style="font-size:12px;font-weight:750;color:#b45309;letter-spacing:.10em;text-transform:uppercase;">DECISION ANALYSIS</div>
  <div class="sp-hero-title" style="font-size:34px;">Cost & Threshold Analysis</div>
  <div class="sp-hero-subtitle">Inspect how operating thresholds change missed failures and false alarms.</div>
</div>
""")

test = pd.read_csv(TABLES_DIR/"final_frozen"/"final_frozen_test_results.csv")
rob = pd.read_csv(TABLES_DIR/"threshold_robustness"/"threshold_robustness_selected.csv")

outputs=st.multiselect(
    "Select outputs to view",
    ["Operating-point metrics","Threshold comparison table","Recall vs precision","False-negative trade-off","Validation threshold selection"],
    default=["Operating-point metrics"],
)

if "Operating-point metrics" in outputs:
    section_title("Operating-point metrics", "Choose one validation-selected operating point and inspect its untouched-test result.")
    point=st.selectbox("Operating point",test["Operating_Point"].tolist(),key="cost_point")
    r=test[test["Operating_Point"].eq(point)].iloc[0]
    c=st.columns(5)
    c[0].metric("Threshold",f"{r.Threshold:.2f}")
    c[1].metric("Precision",f"{r.Precision*100:.2f}%")
    c[2].metric("Recall",f"{r.Recall*100:.2f}%")
    c[3].metric("F1",f"{r.F1*100:.2f}%")
    c[4].metric("False negatives",f"{int(r.FN)}")
    st.caption(f"False positives: {int(r.FP)} · MCC: {r.MCC:.4f} · PR-AUC: {r.PR_AUC*100:.2f}%")

if "Threshold comparison table" in outputs:
    section_title("Threshold comparison", "All frozen-test operating points.")
    cols=["Operating_Point","Threshold","Precision","Recall","F1","MCC","FP","FN","TP","TN"]
    d=test[cols].copy()
    for c in ["Precision","Recall","F1"]: d[c]=(d[c]*100).round(2).astype(str)+"%"
    d["MCC"]=d["MCC"].round(4)
    st.dataframe(d,use_container_width=True,hide_index=True)

if "Recall vs precision" in outputs:
    section_title("Recall vs precision", "Changing the threshold changes the operating point.")
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(test["Recall"]*100,test["Precision"]*100,marker="o")
    for _,r in test.iterrows(): ax.annotate(f"{r.Threshold:.2f}",(r.Recall*100,r.Precision*100),xytext=(5,5),textcoords="offset points",fontsize=9)
    ax.set_xlabel("Recall (%)"); ax.set_ylabel("Precision (%)"); ax.set_title("Precision–recall operating points")
    ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "False-negative trade-off" in outputs:
    section_title("False-negative trade-off", "Compare missed failures with false alarms.")
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.plot(test["Threshold"],test["FN"],marker="o",label="False negatives")
    ax.plot(test["Threshold"],test["FP"],marker="o",label="False positives")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Count"); ax.set_title("FP/FN by operating threshold")
    ax.legend(); ax.grid(alpha=.15); ax.spines[["top","right"]].set_visible(False)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "Validation threshold selection" in outputs:
    section_title("Validation threshold selection", "Thresholds were selected on validation data before final-test evaluation.")
    d=rob.copy()
    d["Cost ratio"]=d["FN_Cost_Ratio"].map(lambda x:f"{int(x)}:1")
    st.dataframe(d[["Cost ratio","Selected_Threshold","Validation_Cost","Precision","Recall","F1","MCC"]].round(4),use_container_width=True,hide_index=True)
    st.caption("These validation costs are selection evidence; the final-test table above is the frozen evaluation.")

render_footer()
