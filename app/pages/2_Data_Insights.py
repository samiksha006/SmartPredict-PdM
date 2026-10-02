import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from app_utils import configure_page, load_dataset, section_title, render_footer

configure_page("SmartPredict-PdM | Data Insights")
df = load_dataset()

st.html("""
<div class="sp-hero">
  <div style="font-size:12px;font-weight:750;color:#2563eb;letter-spacing:.10em;text-transform:uppercase;">DATA INTELLIGENCE</div>
  <div class="sp-hero-title" style="font-size:34px;">Data Insights</div>
  <div class="sp-hero-subtitle">Choose the dataset evidence you want to inspect.</div>
</div>
""")

outputs = st.multiselect(
    "Select outputs to view",
    ["Dataset summary","Data quality","Class distribution","Feature distribution","Correlation heatmap",
     "Failure by machine type","Feature statistics","Sample records"],
    default=["Dataset summary"],
)

if "Dataset summary" in outputs:
    section_title("Dataset summary", "High-level characteristics of the AI4I 2020 dataset.")
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", f"{df.shape[1]}")
    c[2].metric("Failures", f"{int(df['Machine failure'].sum()):,}")
    c[3].metric("Failure rate", f"{df['Machine failure'].mean()*100:.2f}%")

if "Data quality" in outputs:
    section_title("Data quality", "Missing values and duplicate-row checks.")
    quality = pd.DataFrame({
        "Check":["Missing cells","Duplicate rows","Failure records","Normal records"],
        "Value":[int(df.isna().sum().sum()), int(df.duplicated().sum()), int(df["Machine failure"].sum()), int((df["Machine failure"]==0).sum())]
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

if "Class distribution" in outputs:
    section_title("Class distribution", "The failure class is rare in the benchmark dataset.")
    fig, ax = plt.subplots(figsize=(8,4))
    counts=df["Machine failure"].value_counts().sort_index()
    ax.bar(["Normal","Failure"], [counts.get(0,0),counts.get(1,0)])
    ax.set_ylabel("Observations"); ax.set_title("Machine failure distribution")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.15)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "Feature distribution" in outputs:
    section_title("Feature distribution", "Inspect one numeric feature at a time.")
    feature = st.selectbox("Feature", [
        "Air temperature [K]","Process temperature [K]","Rotational speed [rpm]",
        "Torque [Nm]","Tool wear [min]"
    ])
    fig, ax = plt.subplots(figsize=(9,4))
    ax.hist(df[feature].dropna(), bins=30)
    ax.set_xlabel(feature); ax.set_ylabel("Count"); ax.set_title(f"Distribution — {feature}")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.15)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "Correlation heatmap" in outputs:
    section_title("Correlation heatmap", "Numeric feature relationships.")
    numeric = df[["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]","Machine failure"]]
    fig, ax = plt.subplots(figsize=(9,6))
    im=ax.imshow(numeric.corr(), aspect="auto")
    ax.set_xticks(range(len(numeric.columns))); ax.set_xticklabels(numeric.columns,rotation=45,ha="right",fontsize=8)
    ax.set_yticks(range(len(numeric.columns))); ax.set_yticklabels(numeric.columns,fontsize=8)
    fig.colorbar(im,ax=ax,shrink=.8)
    ax.set_title("Numeric correlation matrix")
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "Failure by machine type" in outputs:
    section_title("Failure by machine type", "Failure counts and rates across product types.")
    g=df.groupby("Type")["Machine failure"].agg(["sum","count"])
    g["Failure rate %"]=g["sum"]/g["count"]*100
    st.dataframe(g.rename(columns={"sum":"Failures","count":"Machines"}).reset_index(),use_container_width=True,hide_index=True)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(g.index,g["Failure rate %"]); ax.set_ylabel("Failure rate (%)"); ax.set_title("Failure rate by type")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.15)
    st.pyplot(fig,use_container_width=True); plt.close(fig)

if "Feature statistics" in outputs:
    section_title("Feature statistics", "Descriptive statistics for model input variables.")
    st.dataframe(df[["Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]].describe().T.round(3),use_container_width=True)

if "Sample records" in outputs:
    section_title("Sample records", "Optional preview of raw benchmark observations.")
    n=st.slider("Rows to display",5,50,10,key="data_sample_n")
    st.dataframe(df.head(n),use_container_width=True,hide_index=True)

render_footer()
