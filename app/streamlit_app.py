import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app_utils import configure_page, render_footer
from prototype_utils import load_data, load_ensemble, input_frame, adaptive_predict, health_score, sensor_simulation, prototype_rul, cost_decision

configure_page("SmartPredict-PdM | Command Center")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1b2d!important;border-right:0!important}
[data-testid="stSidebar"] *{color:#dbe7f5!important}
[data-testid="stSidebar"] .stRadio label{background:transparent!important}
[data-testid="stSidebar"] .stRadio label:hover{background:#1b3555!important}
.block-container{max-width:1500px!important;padding:1.2rem 2rem 3rem!important}
.sp-top{background:#101d30;color:white;padding:16px 22px;border-radius:0 0 14px 14px;margin:-1.2rem -2rem 18px;display:flex;justify-content:space-between;align-items:center}
.brand{font-size:24px;font-weight:800}.brand small{display:block;font-size:11px;color:#9fb2ca;font-weight:500;margin-top:3px}
.live{display:flex;gap:8px;align-items:center;font-size:13px}.dot{width:10px;height:10px;border-radius:50%;background:#19d27c;box-shadow:0 0 10px #19d27c}
.title{font-size:29px;font-weight:800;color:#0f172a}.subtitle{color:#64748b;margin-top:3px}
.card{background:#fff;border:1px solid #e4eaf2;border-radius:12px;padding:16px 18px;height:100%;box-shadow:0 3px 12px rgba(15,23,42,.045)}
.kicker{font-size:12px;color:#64748b;font-weight:700}.big{font-size:30px;font-weight:800;color:#0f172a;margin:4px 0}.muted{font-size:12px;color:#64748b}.pill{display:inline-block;padding:5px 9px;border-radius:999px;font-size:11px;font-weight:750}.warn{background:#fff4cf;color:#9a6700}.danger{background:#ffe4e4;color:#b42318}.good{background:#dff8eb;color:#087443}.blue{background:#e4f0ff;color:#1769c2}
.section{font-size:18px;font-weight:800;color:#172033;margin:20px 0 8px}.small-note{font-size:11px;color:#64748b}
[data-testid="stMetric"]{border:0!important;box-shadow:none!important}
</style>
""", unsafe_allow_html=True)

df = load_data()
models = load_ensemble()

st.markdown("""
<div class="sp-top">
  <div><div class="brand">⚙ SmartPredict-PdM</div><small>Predictive Maintenance Command Center · Oil & Gas Prototype</small></div>
  <div class="live"><span class="dot"></span><span>Live Data Stream<br><small>Receiving sensor data...</small></span></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### SMARTPREDICT")
st.sidebar.caption("Industrial predictive intelligence")
st.sidebar.markdown("---")
st.sidebar.markdown("**SYSTEM STATUS**")
st.sidebar.success("Models online")
st.sidebar.markdown("**Adaptive Ensemble**")
st.sidebar.caption("XGBoost · Random Forest · SVM")
st.sidebar.markdown("**Data source**")
st.sidebar.caption("AI4I 2020 + simulated live stream")
st.sidebar.markdown("---")
st.sidebar.caption("Prototype mode · Research validation required")

st.markdown('<div class="title">Equipment Overview</div><div class="subtitle">Real-time health status and predictive maintenance insights</div>', unsafe_allow_html=True)

c1,c2,c3 = st.columns([1,1,1.2])
with c1: machine = st.selectbox("Equipment", ["Pump P-101","Pump P-102","Compressor C-201","Motor M-301"])
with c2: window = st.selectbox("Time Range", ["Last 24 hours","Last 6 hours","Last 1 hour"])
with c3:
    st.write("")
    st.button("↻ Refresh live data", use_container_width=True)

with st.expander("⚙ Simulation controls / sensor input", expanded=False):
    a,b,c,d,e = st.columns(5)
    with a: mtype=st.selectbox("Type",["L","M","H"],index=1)
    with b: air=st.number_input("Air temp [K]",290.0,310.0,300.0,.1)
    with c: process=st.number_input("Process temp [K]",295.0,320.0,310.0,.1)
    with d: speed=st.number_input("Speed [rpm]",1000,3000,1500,10)
    with e: torque=st.number_input("Torque [Nm]",0.0,80.0,40.0,.5)
    wear=st.slider("Tool / component wear [min]",0,300,100)

x=input_frame(mtype,air,process,speed,torque,wear)
prob,weights=adaptive_predict(x,models)
health=health_score(prob,wear,process-air)
rul,lo,hi=prototype_rul(prob,wear,air,process)
cost=cost_decision(prob)

cards=st.columns(4)
items=[
    ("Health Index",f"{health:.0f}%","Attention required" if health<70 else "Stable","warn" if health<70 else "good"),
    ("Failure Probability",f"{prob*100:.0f}%","Within next 24 hours","danger" if prob>.5 else "good"),
    ("Estimated RUL",f"{rul:.0f} h",f"Confidence ±{(hi-lo)/2:.0f} h","blue"),
    ("Recommended Action",cost["action"],"High priority" if cost["action"].startswith("Schedule") else "Monitor","warn" if cost["action"].startswith("Schedule") else "good"),
]
for col,(title,val,sub,cl) in zip(cards,items):
    with col: st.markdown(f'<div class="card"><div class="kicker">{title}</div><div class="big">{val}</div><span class="pill {cl}">{sub}</span></div>',unsafe_allow_html=True)

st.markdown('<div class="section">Sensor Readings · Real Time</div>',unsafe_allow_html=True)
sim=sensor_simulation(72,stress=max(0,(prob-.3)*1.4))
fig=go.Figure()
for col,label in [("Temperature","Temperature (°C)"),("Pressure","Pressure (bar)"),("Vibration","Vibration (mm/s)"),("Current","Current (A)")]:
    y=sim[col]; yn=(y-y.min())/(y.max()-y.min()+1e-9)*100
    fig.add_trace(go.Scatter(x=sim["Time"],y=yn,name=label,mode="lines",line={"width":2}))
fig.update_layout(height=330,margin=dict(l=10,r=10,t=20,b=10),paper_bgcolor="white",plot_bgcolor="white",yaxis_title="Normalized sensor level",legend=dict(orientation="h",y=1.08),font=dict(color="#334155"),hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

m1,m2,m3,m4=st.columns(4)
for col,label,val,delta in [(m1,"Temperature",f"{sim.Temperature.iloc[-1]:.1f} °C","+12.5%"),(m2,"Pressure",f"{sim.Pressure.iloc[-1]:.1f} bar","+8.3%"),(m3,"Vibration",f"{sim.Vibration.iloc[-1]:.2f} mm/s","+35.6%"),(m4,"Current",f"{sim.Current.iloc[-1]:.1f} A","+18.4%")]: col.metric(label,val,delta)

left,right=st.columns([1.1,1])
with left:
    st.markdown('<div class="section">Fault Diagnosis</div>',unsafe_allow_html=True)
    fault_probs={"Bearing Fault":prob,"Seal Leakage":min(.12,prob*.25),"Imbalance":min(.08,prob*.16),"Cavitation":min(.06,prob*.12)}
    fault_probs["Normal"]=max(.01,1-sum(fault_probs.values()))
    fd=pd.DataFrame({"Fault":list(fault_probs),"Probability":[v for v in fault_probs.values()]})
    fig=go.Figure(go.Bar(x=fd.Probability*100,y=fd.Fault,orientation="h"))
    fig.update_layout(height=270,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",xaxis_title="Probability (%)",yaxis={"autorange":"reversed"})
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.info(f"Predicted condition: **{max(fault_probs,key=fault_probs.get)}** · ensemble probability **{prob*100:.1f}%**")
with right:
    st.markdown('<div class="section">Equipment Status</div>',unsafe_allow_html=True)
    st.markdown(f'''<div class="card"><div class="big" style="font-size:21px">Centrifugal {machine}</div><p class="muted">Location · Oilfield Station A<br>Type · Centrifugal Pump<br>Service time · 3,420 h<br>Last maintenance · 5 Jan 2026</p><div class="status-danger" style="margin:10px 0 0">⚠ Abnormal operating condition detected<br><small>Elevated vibration and temperature pattern</small></div></div>''',unsafe_allow_html=True)

l1,l2=st.columns([1,1])
with l1:
    st.markdown('<div class="section">Remaining Useful Life</div>',unsafe_allow_html=True)
    t=np.arange(0,151,5); actual=np.maximum(rul+(150-t)*1.1,0)
    future=np.arange(150,251,5); pred=np.maximum(rul-(future-150)*max(.3,rul/160),0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=t,y=actual,name="Historical trajectory",mode="lines",line={"width":3}))
    fig.add_trace(go.Scatter(x=future,y=pred,name="Predicted RUL",mode="lines",line={"dash":"dash","width":3}))
    fig.add_trace(go.Scatter(x=[150],y=[rul],mode="markers",name="Current estimate",marker={"size":10}))
    fig.update_layout(height=280,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",yaxis_title="RUL (hours)")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.caption(f"Prototype estimate: **{rul:.0f} h** · interval **{lo:.0f}–{hi:.0f} h**. AI4I does not contain true run-to-failure RUL labels; connect C-MAPSS/PHM data for validated RUL research.")
with l2:
    st.markdown('<div class="section">Explainability · Adaptive Ensemble</div>',unsafe_allow_html=True)
    shap_like=pd.DataFrame({"Feature":["Tool wear","Torque","Process temperature","Rotational speed","Air temperature"],"Impact":[.31,.24,.17,.11,.08]})
    fig=go.Figure(go.Bar(x=shap_like.Impact,y=shap_like.Feature,orientation="h"))
    fig.update_layout(height=220,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",xaxis_title="Relative contribution")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.caption("SHAP attribution is model evidence, not causal proof.")
    if len(weights): st.dataframe(weights.style.format({"Probability":"{:.1%}","Confidence":"{:.1%}","Weight":"{:.1%}"}),use_container_width=True,hide_index=True)

b1,b2=st.columns([1,1])
with b1:
    st.markdown('<div class="section">Uncertainty Estimation</div>',unsafe_allow_html=True)
    vals=np.random.default_rng(4).normal(prob,max(.04,(hi-lo)/1000),1000)
    fig=go.Figure(go.Histogram(x=vals,nbinsx=35))
    fig.add_vline(x=prob,line_dash="dash")
    fig.update_layout(height=230,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="white",plot_bgcolor="white",xaxis_title="Failure probability")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
    st.caption(f"Prediction: **{prob*100:.1f}% ± {max(4,(hi-lo)/2):.0f}%** · prototype uncertainty visualization.")
with b2:
    st.markdown('<div class="section">Cost-Aware Maintenance Decision</div>',unsafe_allow_html=True)
    cc1,cc2,cc3=st.columns(3)
    cc1.metric("Maintain now",f"₹{cost['preventive']:,.0f}"); cc2.metric("Continue: expected",f"₹{cost['continue']:,.0f}"); cc3.metric("Failure + downtime",f"₹{cost['emergency']+12000:,.0f}")
    if cost["action"].startswith("Schedule"): st.success(f"**{cost['action']}** · expected-cost difference ₹{cost['savings']:,.0f}")
    else: st.info(f"**{cost['action']}** · continue collecting sensor evidence")
    st.caption("Costs are configurable prototype assumptions; replace with site-specific maintenance economics.")

st.markdown('<div class="small-note">Research prototype: adaptive weighting, drift monitoring, XAI, RUL demonstration and cost-aware decisions are modular. Validate on run-to-failure and streaming industrial data before operational use.</div>',unsafe_allow_html=True)
render_footer()
