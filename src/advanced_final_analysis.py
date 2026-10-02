"""Final error analysis and SHAP analysis for the enhanced frozen model."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix
import shap

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
from config import DATA_PATH,TARGET_COLUMN,RESULTS_DIR
RANDOM_STATE=42
OUT=RESULTS_DIR/"advanced_final_analysis"; OUT.mkdir(parents=True,exist_ok=True)
MODEL=joblib.load(ROOT/"models/final_experiment/advanced_features_scheduled_xgb.joblib")
FEATURES=["Type","Air temperature [K]","Process temperature [K]","Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]
df=pd.read_csv(DATA_PATH); X=df[FEATURES].copy(); y=df[TARGET_COLUMN].astype(int)
X_dev,X_test,y_dev,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=RANDOM_STATE)
X_train,X_val,y_train,y_val=train_test_split(X_dev,y_dev,test_size=.20,stratify=y_dev,random_state=RANDOM_STATE)
p_val=MODEL.predict_proba(X_val)[:,1]
p_test=MODEL.predict_proba(X_test)[:,1]
# Thresholds: model-selected validation MCC plus reference operating points.
val_best=.16
rows=[]
for label,t in [("Default_0.50",.50),("Validation_MCC_0.16",val_best),("Reference_0.11",.11),("Reference_0.05",.05)]:
    pred=(p_test>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
    rows.append({"Operating point":label,"Threshold":t,"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,zero_division=0),"Recall":recall_score(y_test,pred,zero_division=0),"F1":f1_score(y_test,pred,zero_division=0),"ROC_AUC":roc_auc_score(y_test,p_test),"PR_AUC":average_precision_score(y_test,p_test),"MCC":matthews_corrcoef(y_test,pred),"TN":tn,"FP":fp,"FN":fn,"TP":tp})
summary=pd.DataFrame(rows); summary.to_csv(OUT/"threshold_error_summary.csv",index=False)
records=X_test.copy(); records["Actual Failure"]=y_test.values; records["Failure Probability"]=p_test; records["Default Prediction"]=(p_test>=.50).astype(int); records["Validation Threshold Prediction"]=(p_test>=.16).astype(int); records.to_csv(OUT/"final_test_probabilities.csv",index=False)
# SHAP on 1000 deterministic test samples.
model_input=MODEL.named_steps["feature_engineering"].transform(X_test.iloc[:1000])
pre=MODEL.named_steps["preprocessor"]; xgb=MODEL.named_steps["xgb"]
trans=pre.transform(model_input); trans=trans.toarray() if hasattr(trans,'toarray') else trans
names=[str(n).replace('num__','').replace('cat__','') for n in pre.get_feature_names_out()]
sv=shap.TreeExplainer(xgb).shap_values(trans)
sv=np.asarray(sv)
imp=pd.DataFrame({"Feature":names,"Mean_abs_SHAP":np.abs(sv).mean(axis=0)}).sort_values("Mean_abs_SHAP",ascending=False)
imp.to_csv(OUT/"shap_feature_importance.csv",index=False)
print(summary.to_string(index=False)); print('\nTop SHAP features:'); print(imp.head(10).to_string(index=False))
