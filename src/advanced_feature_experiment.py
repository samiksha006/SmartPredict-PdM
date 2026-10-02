"""Frozen evaluation of XGBoost with domain-derived features, LR scheduling and early stopping."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from xgboost.callback import LearningRateScheduler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PATH, TARGET_COLUMN, MODELS_DIR, RESULTS_TABLE_DIR

RANDOM_STATE = 42
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 6400, 1600, 2000
ORIGINAL_NUMERIC = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]"
]
CATEGORICAL = ["Type"]
BASE_FEATURES = CATEGORICAL + ORIGINAL_NUMERIC
DERIVED_FEATURES = [
    "Temperature Difference [K]", "Temperature Ratio", "Mechanical Power Proxy",
    "Torque per Speed", "Wear per Speed", "Thermal-Wear Interaction"
]
ALL_NUMERIC = ORIGINAL_NUMERIC + DERIVED_FEATURES

OUT_DIR = RESULTS_TABLE_DIR / "final_experiment"
MODEL_DIR = MODELS_DIR / "final_experiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


from model_utils import add_advanced_features, identity_features, lr_schedule

def make_preprocessor(numeric_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])



def make_model():
    return XGBClassifier(
        n_estimators=800, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=1,
        gamma=0.1, reg_alpha=0.0, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
        early_stopping_rounds=60,
    )


def threshold_by_mcc(y, p):
    best = None
    for t in np.arange(0.01, 0.501, 0.01):
        pred = (p >= t).astype(int)
        score = matthews_corrcoef(y, pred)
        candidate = (score, -t, t)
        if best is None or candidate > best[0]:
            best = (candidate, score, t)
    return best[2], best[1]


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "MCC": matthews_corrcoef(y, pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Threshold": threshold,
    }


def run_variant(X_train, y_train, X_val, y_val, X_test, y_test, engineered):
    if engineered:
        tr = add_advanced_features(X_train)
        va = add_advanced_features(X_val)
        te = add_advanced_features(X_test)
        numeric = ALL_NUMERIC
        name = "advanced_features_scheduled_xgb"
    else:
        tr, va, te = X_train.copy(), X_val.copy(), X_test.copy()
        numeric = ORIGINAL_NUMERIC
        name = "baseline_scheduled_xgb"

    pre = make_preprocessor(numeric)
    tr_t = pre.fit_transform(tr)
    va_t = pre.transform(va)
    te_t = pre.transform(te)

    model = make_model()
    model.set_params(callbacks=[LearningRateScheduler(lr_schedule)])
    model.fit(tr_t, y_train, eval_set=[(va_t, y_val)], verbose=False)

    p_val = model.predict_proba(va_t)[:,1]
    threshold, val_mcc = threshold_by_mcc(y_val, p_val)
    p_test = model.predict_proba(te_t)[:,1]
    result = metrics(y_test, p_test, threshold)
    result.update({"Model": name, "Best_iteration": int(model.best_iteration),
                   "Validation_MCC": float(val_mcc), "Training_samples": len(tr)})

    # Assemble a prediction-ready pipeline from already-fitted components.
    from sklearn.preprocessing import FunctionTransformer
    feature_step = FunctionTransformer(add_advanced_features, validate=False) if engineered else FunctionTransformer(identity_features, validate=False)
    pipeline = Pipeline([("feature_engineering", feature_step), ("preprocessor", pre), ("xgb", model)])
    dump(pipeline, MODEL_DIR / f"{name}.joblib")
    return result, pipeline


def main():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COLUMN].astype(int)
    X = df[BASE_FEATURES].copy()

    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20,
                                                      stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.20,
                                                       stratify=y_dev, random_state=RANDOM_STATE)
    assert (len(X_train), len(X_val), len(X_test)) == (TRAIN_SIZE, VAL_SIZE, TEST_SIZE)

    results = []
    for engineered in (False, True):
        r, _ = run_variant(X_train, y_train, X_val, y_val, X_test, y_test, engineered)
        results.append(r)

    table = pd.DataFrame(results)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC", "MCC",
            "TN", "FP", "FN", "TP", "Threshold", "Best_iteration", "Validation_MCC", "Training_samples"]
    table = table[cols]
    table.to_csv(OUT_DIR / "advanced_feature_final_test.csv", index=False)

    inventory = pd.DataFrame({"Feature": DERIVED_FEATURES,
                              "Definition": [
                                  "Process temperature - Air temperature",
                                  "Process temperature / Air temperature",
                                  "Torque * Rotational speed",
                                  "Torque / Rotational speed",
                                  "Tool wear / Rotational speed",
                                  "(Process temperature - Air temperature) * Tool wear",
                              ]})
    inventory.to_csv(OUT_DIR / "advanced_feature_inventory.csv", index=False)
    metadata = {
        "protocol": "Frozen 6400 train / 1600 validation / 2000 final test",
        "random_state": RANDOM_STATE,
        "objective": "binary classification; evaluation logloss",
        "initial_learning_rate": 0.08,
        "learning_rate_schedule": "max(0.01, 0.08 * 0.985^epoch)",
        "max_boosting_rounds": 800,
        "early_stopping_rounds": 60,
        "threshold_selection": "maximum validation MCC over 0.01-0.50",
        "derived_features": DERIVED_FEATURES,
        "test_set_used_for_tuning": False,
    }
    (OUT_DIR / "advanced_feature_training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(table.to_string(index=False))
    print("\nSaved models:", MODEL_DIR)

if __name__ == "__main__":
    main()
