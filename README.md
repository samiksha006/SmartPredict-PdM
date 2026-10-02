# SmartPredict-PdM

**An academic predictive-maintenance prototype for imbalanced machine-failure classification**

> **Research prototype:** This project is for coursework and experimentation. Its predictions, simulated sensor stream, prototype remaining-useful-life (RUL) estimate, and maintenance recommendations are not validated for operational or safety-critical use.

## Problem statement

Unexpected equipment failures cause downtime and maintenance expense. Predictive-maintenance classifiers can help identify failure-prone operating conditions, but failures are rare and missed failures may be more costly than false alarms. This project studies the trade-offs among kernel classifiers and tree-based ensembles on the AI4I 2020 benchmark, including class-imbalance handling, validation-selected thresholds, error analysis, and post-hoc explanations.

## Objective

Compare baseline and tuned classifiers for the AI4I machine-failure label, assess precision/recall and ranking performance under class imbalance, and demonstrate how a decision threshold can be changed to reflect different false-negative/false-positive cost assumptions. The experiments do not establish that the models generalize to real equipment or future time periods.

## Dataset

The project includes the AI4I 2020 Predictive Maintenance Dataset (10,000 rows; 14 columns), downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). The bundled `data/ai4i2020.csv` and source archive contain the same dataset. UCI describes the data as synthetic machine operating data with product type, air/process temperatures, rotational speed, torque, tool wear, a machine-failure label, and five failure-mode indicators.

The final experiment metadata records 339 failures in the 10,000-row dataset (3.39%). The failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from the primary predictor set because they are closely coupled to the target and would leak failure information in a pre-failure setting. `UDI` and `Product ID` are also excluded as identifiers. Retain attribution and follow the usage terms on the UCI dataset page when redistributing the bundled data.

## Methodology

- Stratified splits are used in the experiment scripts. The master experiment records 6,400 training, 1,600 validation, and 2,000 test rows using seed 42. Other scripts intentionally contain their own protocols (for example, an independent 80/20 final-evaluation split); their tables must not be treated as a single pooled experiment.
- Preprocessing imputes numeric values with the median, scales numeric features, imputes categorical values with the most frequent category, and one-hot encodes `Type`.
- Thresholds are selected from validation data in the master/threshold experiments before reporting their corresponding test results. Do not tune thresholds on the held-out test data.
- Rebalancing is performed inside the training pipeline for the SMOTE and SMOTE-Tomek experiments. The test set is not resampled.
- Random seeds are explicitly set in the scripts where applicable. Results in `results/` are existing project outputs; running scripts may overwrite files in those folders.

## Models implemented

The code and saved artifacts cover Logistic Regression, Decision Tree, K-Nearest Neighbors, linear/polynomial/RBF SVMs, Random Forest, Extra Trees, AdaBoost, XGBoost, soft voting, and stacking. Separate experiments evaluate class weighting, SMOTE, and SMOTE-Tomek. Not every model appears in every experiment or result table.

## Feature engineering

The primary pipeline uses five numeric measurements plus categorical product `Type`. An advanced-feature experiment adds:

- Temperature difference and temperature ratio
- Mechanical power proxy (`Torque × rotational speed`)
- Torque per speed and wear per speed
- Thermal-wear interaction

Feature definitions are implemented in `src/model_utils.py`. Engineered features are experimental proxies, not validated physical health indicators.

## Hyperparameter optimization

`src/optimize_xgboost.py` uses Optuna with 30 trials and 3-fold stratified cross-validation. Search history and summary tables are saved under `results/tables/`. The recorded `optimized_xgboost_results.csv` reports PR-AUC 0.7929 against its recorded baseline PR-AUC 0.8057 (difference -0.0128); optimization therefore did **not** improve that particular recorded comparison. Other result files come from separate runs/protocols and are reported separately rather than used to claim a universal improvement.

## Evaluation metrics

The project reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC (average precision), Matthews correlation coefficient (MCC), confusion-matrix counts, calibration measures, and threshold-dependent error/cost summaries. PR-AUC, recall, MCC, and false-negative counts are useful complements to accuracy for this imbalanced label.

## Recorded final results

The following values are transcribed from `results/tables/final_test_results.csv` (the script’s stratified 80/20 holdout protocol, seed 42). These are the saved project results, not a claim that this README run reproduced them.

| Model | Precision | Recall | F1 | PR-AUC | MCC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost Baseline | 0.9583 | 0.6765 | 0.7931 | 0.8464 | 0.7998 | 1930 / 2 / 22 / 46 |
| Optimized XGBoost | 0.9074 | 0.7206 | 0.8033 | 0.8462 | 0.8028 | 1927 / 5 / 19 / 49 |
| XGBoost + SMOTE-Tomek | 0.4328 | 0.8529 | 0.5743 | 0.8103 | 0.5897 | 1856 / 76 / 10 / 58 |

These rows illustrate different operating trade-offs: SMOTE-Tomek reduces missed failures in this table while producing many more false alarms. The separate 64/16/20 master experiment and advanced-feature experiment are saved under `results/tables/final_experiment/`; do not compare their values as though they shared an identical evaluation protocol.

## Ablation study

`results/tables/ablation/ablation_results.csv` records the following thresholded test metrics for its own stratified split and validation-based threshold selection:

| Stage | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline | 0.8125 | 0.7647 | 0.7879 |
| Optimized | 0.8500 | 0.7500 | 0.7969 |
| Class weight | 0.9048 | 0.5588 | 0.6909 |
| SMOTE | 0.7846 | 0.7500 | 0.7669 |
| SMOTE-Tomek | 0.8033 | 0.7206 | 0.7597 |

The saved ablation does not show a single strategy dominating every metric. Interpret it with its confusion counts and threshold tables, not accuracy alone.

## Error analysis

The optimized-XGBoost row in the saved final-test table has 19 false negatives and 5 false positives (24 errors total). `results/tables/error_analysis_summary.csv`, `results/final_error_analysis/`, and its figures provide additional per-case and threshold-specific error breakdowns. A missed failure and a false alarm have different operational consequences; their relative cost is a scenario assumption, not a measured business cost.

## SHAP and explainability

The project includes global, direction, missed-failure, rescued-failure, and individual SHAP analyses. Saved visualizations and tables are under `results/final_shap_analysis/` and `results/advanced_final_analysis/`. In the advanced-feature importance table, the leading mean absolute SHAP entries are Mechanical Power Proxy (0.7120), Tool wear (0.6646), and Rotational speed (0.5107). These are model-specific attributions, not causal effects or guarantees of physical importance.

## Cost-aware threshold analysis

`results/tables/cost_sensitive/cost_sensitive_test_results.csv` evaluates threshold choices under illustrative false-negative-to-false-positive cost ratios. In that saved table, the 1:1 row uses threshold 0.35 (17 FN, 9 FP); the 5:1 row uses 0.35 (17 FN, 9 FP); and the 10:1 row uses 0.05 (10 FN, 68 FP). This demonstrates the recall/false-alarm trade-off under the project’s chosen cost assumptions. The ratios and resulting costs are analytical scenarios, not estimated maintenance costs.

## Streamlit application

Run the prototype dashboard with `streamlit run app/streamlit_app.py`. It provides equipment overview, failure prediction, data insights, model performance, cost/threshold analysis, error analysis, SHAP explainability, live-monitoring replay, fault-diagnosis, a prototype RUL view, drift monitoring, and maintenance-decision views.

The sensor stream is simulated/replayed. AI4I is not a run-to-failure time-series dataset and has no genuine RUL labels; the RUL display is explicitly a prototype estimator. The dashboard’s “next 24 hours” wording and maintenance outputs are illustrative UI behavior, not a validated prediction horizon or deployment recommendation.

## Installation

Use Python 3.10 or newer and install the declared dependencies in an isolated virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For Linux/macOS, create and activate the environment with `python3 -m venv .venv` and `source .venv/bin/activate`, then run the two pip commands above.

## Running the project

From the repository root:

```bash
# Unit tests
python -m pytest

# Exploratory notebook
jupyter notebook notebooks/01_EDA.ipynb

# Streamlit interface
streamlit run app/streamlit_app.py

# Example analysis/experiment entry points
python src/eda_analysis.py
python src/train_pipeline.py
python src/master_experiment.py
```

Other experiment scripts under `src/` implement kernel comparisons, Optuna optimization, imbalance/ablation studies, threshold and cost analyses, repeated cross-validation, error analysis, SHAP analysis, and reproducibility audits. Review a script’s split and output paths before rerunning it; outputs can be overwritten.

## Project structure

```text
SmartPredict-PdM/
├── app/                 # Streamlit app, utilities, and multipage views
├── data/                # UCI AI4I CSV and original source archive
├── models/              # Serialized experiment models (one large model uses Git LFS)
├── notebooks/           # EDA notebook
├── results/
│   ├── figures/         # EDA, kernel, model, threshold, and analysis plots
│   ├── tables/          # Metrics, thresholds, ablations, CV, and metadata
│   ├── advanced_final_analysis/
│   ├── final_error_analysis/
│   └── final_shap_analysis/
├── src/                 # Data, preprocessing, training, and experiment scripts
├── tests/               # Dataset loading and target sanity tests
├── .gitattributes        # Git LFS rule for the largest model artifact
├── .gitignore
└── requirements.txt
```

`models/` contains checked-in fitted artifacts used by the prototype and/or retained for reproducibility. `models/extra_trees.joblib` is stored through Git LFS. Install Git LFS before cloning if you need the model contents:

```bash
git lfs install
git clone https://github.com/samiksha006/SmartPredict-PdM.git
git lfs pull
```

## Limitations

- AI4I is synthetic, cross-sectional benchmark data; random stratified splits do not test temporal or machine-to-machine generalization.
- Failure-mode columns encode information tied to the target and are excluded from the primary predictors. Any experiment changing that feature policy must document leakage risk.
- The data does not contain genuine run-to-failure histories or RUL labels. The dashboard’s RUL, simulated sensor feed, equipment/fault labels, and maintenance actions are demonstrations only.
- Cost ratios are user-selected scenarios, not costs validated with operators or maintenance records.
- Thresholds and model performance can change across scripts because their split, features, fitting, and threshold-selection protocols differ.
- Dependency requirements are lower bounds, not a fully locked environment; package-version changes can affect reproducibility.

## Reproducibility notes

The checked-in CSVs, figures, notebook, source, and model files are retained project artifacts. Main experiment scripts use explicit seeds (commonly 42), but not all scripts share the same split or fitted estimator. Use each result’s producing script and metadata when interpreting it, preserve the held-out test set, and rerun experiments only when you intend to regenerate outputs. No new experimental result is claimed by this documentation.
