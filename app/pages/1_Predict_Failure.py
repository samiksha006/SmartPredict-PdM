import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from app_utils import configure_page, load_model, create_input_dataframe, predict_failure, get_shap_explanation, THRESHOLDS, render_footer

configure_page("SmartPredict-PdM | Predict Failure")

st.html("""
<div class="sp-hero">
  <div style="font-size:12px;font-weight:750;color:#2563eb;letter-spacing:.10em;text-transform:uppercase;">PREDICTIVE MAINTENANCE</div>
  <div class="sp-hero-title" style="font-size:34px;">Machine Failure Prediction</div>
  <div class="sp-hero-subtitle">Enter operating conditions, choose a decision strategy, and inspect only the outputs you need.</div>
</div>
""")

model=load_model()

st.subheader("1 · Machine operating conditions")
left,right=st.columns(2)
with left:
    machine_type=st.selectbox("Machine Type",["L","M","H"])
    air_temperature=st.number_input("Air Temperature [K]",290.0,310.0,300.0,0.1)
    process_temperature=st.number_input("Process Temperature [K]",295.0,320.0,310.0,0.1)
with right:
    rotational_speed=st.number_input("Rotational Speed [rpm]",1000,3000,1500,10)
    torque=st.number_input("Torque [Nm]",0.0,80.0,40.0,0.5)
    tool_wear=st.number_input("Tool Wear [min]",0,300,100,1)

st.subheader("2 · Decision strategy")
threshold_label=st.selectbox("Operating strategy",list(THRESHOLDS.keys()))
threshold=THRESHOLDS[threshold_label]
st.caption(f"Selected threshold: {threshold:.2f}. The threshold controls when probability becomes a failure decision.")

input_df=create_input_dataframe(machine_type,air_temperature,process_temperature,rotational_speed,torque,tool_wear)

if st.button("Analyze Machine",type="primary",use_container_width=True):
    prediction,probability=predict_failure(model,input_df,threshold)
    st.session_state.update(prediction=prediction,probability=probability,input_df=input_df,threshold=threshold)

if "prediction" not in st.session_state:
    st.info("Enter the operating conditions and select **Analyze Machine** to generate a prediction.")
else:
    prediction=st.session_state["prediction"]; probability=st.session_state["probability"]; threshold=st.session_state["threshold"]
    st.subheader("3 · Prediction output")
    if prediction:
        st.error("FAILURE RISK DETECTED")
    else:
        st.success("NO FAILURE SIGNAL DETECTED")
    outputs=st.multiselect("Select prediction outputs",["Probability","Decision threshold","Input values","Probability bar","SHAP explanation"],default=["Probability","Decision threshold"])
    if "Probability" in outputs:
        st.metric("Estimated failure probability",f"{probability*100:.2f}%")
    if "Decision threshold" in outputs:
        st.metric("Decision threshold",f"{threshold:.2f}")
    if "Input values" in outputs:
        st.dataframe(st.session_state["input_df"],use_container_width=True,hide_index=True)
    if "Probability bar" in outputs:
        st.progress(float(np.clip(probability,0,1)),text=f"Failure probability: {probability*100:.2f}%")
        st.caption(f"Decision threshold: {threshold:.2f} ({threshold*100:.1f}%).")
    if "SHAP explanation" in outputs:
        st.info("SHAP values explain model behaviour for this input; they are not causal effects.")
        try:
            _,names,values=get_shap_explanation(model,st.session_state["input_df"])
            order=np.argsort(np.abs(values))[::-1][:6][::-1]
            fig,ax=plt.subplots(figsize=(9,4.5))
            ax.barh([names[i] for i in order],[values[i] for i in order])
            ax.axvline(0,linewidth=1); ax.set_xlabel("SHAP contribution"); ax.set_title("Features influencing this prediction")
            ax.spines[["top","right"]].set_visible(False); ax.grid(axis="x",alpha=.15)
            st.pyplot(fig,use_container_width=True); plt.close(fig)
        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

render_footer()
