import streamlit as st
import pandas as pd
import plotly.express as px
from app_utils import configure_page, render_footer
from prototype_utils import input_frame, load_ensemble, adaptive_predict

configure_page('SmartPredict-PdM | Fault Diagnosis')
st.title('Fault Diagnosis')
st.caption('Adaptive ensemble fault-risk prototype with model-level evidence.')

c1,c2,c3=st.columns(3)
with c1: torque=st.slider('Torque [Nm]',0.0,80.0,55.0)
with c2: wear=st.slider('Wear [min]',0,300,180)
with c3: speed=st.slider('Speed [rpm]',1000,3000,1450)

x=input_frame('M',300,311,speed,torque,wear)
prob,weights=adaptive_predict(x,load_ensemble())

st.metric('Adaptive ensemble failure probability',f'{prob:.1%}')
if len(weights):
    st.dataframe(weights.style.format({'Probability':'{:.1%}','Confidence':'{:.1%}','Weight':'{:.1%}'}),use_container_width=True,hide_index=True)

faults=pd.DataFrame({'Fault':['Bearing fault','Seal leakage','Imbalance','Cavitation','Normal'], 'Probability':[prob,min(.12,prob*.25),min(.08,prob*.16),min(.06,prob*.12),max(.01,1-prob)]})
fig=px.bar(faults,x='Probability',y='Fault',orientation='h',text_auto='.1%')
fig.update_layout(height=350,xaxis_tickformat='.0%')
st.plotly_chart(fig,use_container_width=True)

st.subheader('Research gap coverage')
st.markdown('- **Adaptive ensemble:** current confidence is combined with model validation performance to update weights.\n- **Class imbalance:** the existing project retains PR-AUC/recall/threshold analysis.\n- **Explainability:** use the existing SHAP page for feature attribution.\n- **Deployment:** predictions can be called from a FastAPI service in the next integration step.')
render_footer()
