import streamlit as st
import plotly.express as px
import pandas as pd
from app_utils import configure_page, render_footer
from prototype_utils import input_frame, adaptive_predict, load_ensemble, cost_decision

configure_page('SmartPredict-PdM | Maintenance Decision')
st.title('Cost-Aware Maintenance Decision')
st.caption('Convert model risk into an economic comparison. All costs below are editable assumptions.')

c1,c2,c3=st.columns(3)
with c1: probability=st.slider('Failure probability',0.0,1.0,0.78,0.01)
with c2: preventive=st.number_input('Preventive maintenance cost ₹',1000,100000,5000,500)
with c3: downtime=st.number_input('Downtime cost ₹',1000,200000,12000,1000)
emergency=st.number_input('Emergency repair cost ₹',1000,300000,18000,1000)

res=cost_decision(probability,preventive,downtime,emergency)
rows=pd.DataFrame({'Scenario':['Preventive maintenance now','Continue operation (expected)','Emergency failure exposure'], 'Expected Cost':[res['preventive'],res['continue'],res['emergency']+downtime]})
fig=px.bar(rows,x='Scenario',y='Expected Cost',text_auto='₹,.0f')
fig.update_layout(height=380,yaxis_title='Cost (₹)')
st.plotly_chart(fig,use_container_width=True)

st.metric('Decision',res['action'])
if res['action'].startswith('Schedule'): st.success(f"Preventive action has lower modeled expected cost by ₹{res['savings']:,.0f}.")
else: st.info('Under the supplied assumptions, continued monitoring has lower modeled expected cost. Recalculate when new sensor evidence arrives.')

st.subheader('Decision trace')
st.write({'Failure probability':f'{probability:.1%}','Preventive cost':f"₹{preventive:,.0f}",'Expected continue cost':f"₹{res['continue']:,.0f}",'Emergency + downtime':f"₹{res['emergency']+downtime:,.0f}"})
st.caption('This is a decision-support calculation, not an autonomous maintenance authorization. Site-specific costs and safety constraints must be supplied by the operator.')
render_footer()
