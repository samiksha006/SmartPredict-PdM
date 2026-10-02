import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from app_utils import configure_page, load_dataset, render_footer
from prototype_utils import psi

configure_page('SmartPredict-PdM | Drift Monitoring')
st.title('Concept Drift Monitoring')
st.caption('Detect distribution changes before automatically retraining a production model.')
df=load_dataset()
feature=st.selectbox('Feature to monitor',['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]'])
split=st.slider('Baseline percentage',50,80,70)
base=df[feature].iloc[:int(len(df)*split/100)].dropna().values
current=df[feature].iloc[int(len(df)*split/100):].dropna().values
score=psi(base,current)
status='Drift detected' if score>=0.20 else ('Watch' if score>=0.10 else 'Stable')
c1,c2,c3=st.columns(3); c1.metric('PSI',f'{score:.3f}'); c2.metric('Status',status); c3.metric('Baseline samples',f'{len(base):,}')
plot=pd.DataFrame({'Value':np.r_[base,current],'Window':['Baseline']*len(base)+['Current']*len(current)})
fig=px.histogram(plot,x='Value',color='Window',nbins=30,barmode='overlay',opacity=.65)
fig.update_layout(height=400)
st.plotly_chart(fig,use_container_width=True)
st.subheader('Retraining policy')
st.code('No drift → continue\nWatch → increase monitoring frequency\nDrift detected → train candidate → validate → deploy only if candidate passes acceptance tests',language='text')
st.info('The AI4I rows are not a true production time stream. This page demonstrates the drift-monitoring mechanism and should be validated on chronologically collected sensor data.')
render_footer()
