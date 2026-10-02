import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import (
    DATA_PATH, TARGET, RANDOM_STATE, TEST_SIZE,
    MODEL_DIR, TABLES_DIR
)
from data_utils import load_data
from preprocessing import build_feature_frame, build_preprocessor
from models import get_models
from evaluation import evaluate_model

def main():
    df = load_data(DATA_PATH)

    X = build_feature_frame(df)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    models = get_models(RANDOM_STATE)

    results = []
    fitted = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        print(f"Training: {name}")
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(pipe, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        fitted[name] = pipe

    results_df = pd.DataFrame(results)
    results_df = results_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1",
         "ROC_AUC", "PR_AUC", "MCC", "TN", "FP", "FN", "TP"]
    ]

    results_df.to_csv(TABLES_DIR / "baseline_model_results.csv", index=False)

    # Save models
    for name, model in fitted.items():
        filename = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / filename)

    print("\nResults:")
    print(results_df.sort_values("PR_AUC", ascending=False).to_string(index=False))
    print(f"\nSaved results to: {TABLES_DIR / 'baseline_model_results.csv'}")

if __name__ == "__main__":
    main()
