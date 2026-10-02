
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ai4i2020.csv"
MODELS = ROOT / "models"
TABLES = ROOT / "results" / "tables"

FEATURES = [
    "Type","Air temperature [K]","Process temperature [K]",
    "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"
]
NUMERIC = FEATURES[1:]

@st.cache_data
def load_data():
    return pd.read_csv(DATA)

@st.cache_resource
def load_ensemble():
    candidates = {
        "Optimized XGBoost": MODELS/"optimized_xgboost_final.joblib",
        "Random Forest": MODELS/"random_forest_final.joblib",
        "Polynomial SVM": MODELS/"polynomial_svm_optimized.joblib",
        "Linear SVM": MODELS/"linear_svm_optimized.joblib",
    }
    out={}
    for name,path in candidates.items():
        if path.exists():
            try: out[name]=joblib.load(path)
            except Exception: pass
    return out

def input_frame(machine_type="M", air=300.0, process=310.0, speed=1500, torque=40.0, wear=100):
    return pd.DataFrame([{
        "Type":machine_type, "Air temperature [K]":air,
        "Process temperature [K]":process, "Rotational speed [rpm]":speed,
        "Torque [Nm]":torque, "Tool wear [min]":wear
    }])

def adaptive_predict(x, models=None):
    models = models or load_ensemble()
    rows=[]
    for name,model in models.items():
        try:
            p=float(model.predict_proba(x)[0,1])
            rows.append((name,p))
        except Exception:
            continue
    if not rows:
        return 0.0, pd.DataFrame(columns=["Model","Probability","Confidence","Weight"])
    # Prototype adaptive weighting: historical validation F1 × current confidence.
    perf = {"Optimized XGBoost":0.7731, "Random Forest":0.7420, "Polynomial SVM":0.70, "Linear SVM":0.68}
    data=[]
    for name,p in rows:
        confidence=max(p,1-p)
        weight=perf.get(name,0.65)*(0.55+0.45*confidence)
        data.append([name,p,confidence,weight])
    d=pd.DataFrame(data,columns=["Model","Probability","Confidence","RawWeight"])
    d["Weight"]=d["RawWeight"]/d["RawWeight"].sum()
    ensemble=float((d["Probability"]*d["Weight"]).sum())
    return ensemble,d[["Model","Probability","Confidence","Weight"]]

def health_score(probability, wear=100, temp_delta=10, vibration_proxy=1.5):
    risk = 0.65*probability + 0.20*np.clip(wear/300,0,1) + 0.10*np.clip(abs(temp_delta)/20,0,1) + 0.05*np.clip(vibration_proxy/8,0,1)
    return float(np.clip(100*(1-risk),0,100))

def sensor_simulation(n=72, seed=7, stress=0.0):
    rng=np.random.default_rng(seed)
    t=pd.date_range("2026-09-24 14:00", periods=n, freq="5min")
    phase=np.linspace(0,1,n)
    temp=58+7*phase+rng.normal(0,1.2,n)+stress*8*phase
    pressure=108+22*phase+rng.normal(0,2,n)+stress*20*phase
    vibration=1.8+1.5*phase+rng.normal(0,.18,n)+stress*2.2*phase
    current=18+5*phase+rng.normal(0,.5,n)+stress*6*phase
    return pd.DataFrame({"Time":t,"Temperature":temp,"Pressure":pressure,"Vibration":vibration,"Current":current})

def prototype_rul(probability, wear, temp, process, hours_cap=500):
    # Demonstration estimator only: AI4I has no run-to-failure trajectories.
    wear_factor=np.clip(wear/250,0,1)
    thermal=np.clip((process-temp-5)/15,0,1)
    risk=np.clip(.60*probability+.25*wear_factor+.15*thermal,0,1)
    rul=float(np.clip(hours_cap*(1-risk),6,hours_cap))
    uncertainty=max(6, rul*0.16)
    return rul, max(0,rul-uncertainty), min(hours_cap,rul+uncertainty)

def psi(expected, actual, bins=10):
    expected=np.asarray(expected,float); actual=np.asarray(actual,float)
    qs=np.linspace(0,1,bins+1)
    edges=np.unique(np.quantile(expected,qs))
    if len(edges)<3: return 0.0
    e=np.histogram(expected,bins=edges)[0]/len(expected)
    a=np.histogram(actual,bins=edges)[0]/len(actual)
    e=np.clip(e,1e-4,1); a=np.clip(a,1e-4,1)
    return float(np.sum((a-e)*np.log(a/e)))

def cost_decision(probability, preventive=5000, downtime=12000, emergency=18000, delay_hours=24):
    expected_failure=probability*(emergency+downtime)
    continue_cost=expected_failure
    preventive_cost=preventive
    if continue_cost > preventive_cost:
        action="Schedule Maintenance"
    else:
        action="Continue Monitoring"
    return {"preventive":preventive_cost,"continue":continue_cost,"emergency":emergency,
            "action":action,"savings":abs(continue_cost-preventive_cost)}

def model_performance():
    path=TABLES/"ensemble_model_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()
