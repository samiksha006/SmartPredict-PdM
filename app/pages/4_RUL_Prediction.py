import streamlit as st
import numpy as np
import plotly.graph_objects as go
from app_utils import configure_page, render_footer
from prototype_utils import input_frame, adaptive_predict, prototype_rul, load_ensemble

configure_page('SmartPredict-PdM | RUL Prediction')
st.title('Remaining Useful Life (RUL)')
st.caption('Prototype RUL interface. The bundled AI4I dataset does not contain genuine run-to-failure trajectories.')

c1,c2,c3=st.columns(3)
with c1: wear=st.slider('Current wear [min]',0,300,150)
with c2: temp=st.slider('Air temperature [K]',290.0,310.0,300.0,.1)
with c3: process=st.slider('Process temperature [K]',295.0,320.0,310.0,.1)

x=input_frame('M',temp,process,1500,45,wear)
prob,_=adaptive_predict(x,load_ensemble())
rul,lo,hi=prototype_rul(prob,wear,temp,process)
cols=st.columns(3); cols[0].metric('Estimated RUL',f'{rul:.0f} h'); cols[1].metric('Lower bound',f'{lo:.0f} h'); cols[2].metric('Upper bound',f'{hi:.0f} h')

t=np.arange(0,250,5); current=150
hist=np.maximum(rul+(current-t)*1.1,0); future=np.maximum(rul-(t-current)*max(.25,rul/180),0)
fig=go.Figure(); fig.add_trace(go.Scatter(x=t[t<=current],y=hist[t<=current],name='Historical/degradation proxy')); fig.add_trace(go.Scatter(x=t[t>=current],y=future[t>=current],name='Predicted RUL',line={'dash':'dash'})); fig.add_vline(x=current,line_dash='dot')
fig.update_layout(height=450,xaxis_title='Operating timeline (hours)',yaxis_title='RUL (hours)')
st.plotly_chart(fig,use_container_width=True)

st.warning('For a research-valid RUL result, train this module on NASA C-MAPSS, IMS, FEMTO-ST, or another run-to-failure dataset. The current values are explicitly a prototype estimator, not a measured RUL ground truth.')
render_footer()
