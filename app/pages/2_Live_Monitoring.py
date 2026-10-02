import streamlit as st
import plotly.graph_objects as go
from app_utils import configure_page, render_footer
from prototype_utils import sensor_simulation

configure_page('SmartPredict-PdM | Live Monitoring')
st.title('Live Monitoring')
st.caption('Streaming-style visualization using replayed/simulated sensor data. Replace the simulator with MQTT/Kafka in deployment.')

stress=st.slider('Operating stress',0.0,1.0,0.35,0.05)
sim=sensor_simulation(144,stress=stress)
fig=go.Figure()
for c in ['Temperature','Pressure','Vibration','Current']:
    fig.add_trace(go.Scatter(x=sim.Time,y=sim[c],name=c,mode='lines'))
fig.update_layout(height=500,hovermode='x unified',xaxis_title='Time',yaxis_title='Sensor value')
st.plotly_chart(fig,use_container_width=True)

cols=st.columns(4)
for col,c in zip(cols,['Temperature','Pressure','Vibration','Current']):
    col.metric(c,f'{sim[c].iloc[-1]:.2f}',f'{sim[c].iloc[-1]-sim[c].iloc[-6]:+.2f}')

st.subheader('Streaming architecture')
st.code('Sensors → MQTT/Kafka → Feature Service → Adaptive Ensemble → Alert Engine → Dashboard → Maintenance Log',language='text')
st.info('Research note: this page demonstrates the deployment interface. The bundled prototype uses deterministic simulation so it runs without external hardware.')
render_footer()
