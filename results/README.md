# Experiment outputs

This directory contains the project’s saved result tables and figures. They are retained as they existed in the project; no values were edited or regenerated while preparing the repository.

Outputs are grouped by analysis where possible:

- `tables/`: baseline/final evaluations, model comparisons, ablations, threshold/cost analysis, repeated cross-validation, and experiment metadata.
- `figures/`: EDA, kernel-machine comparisons, model performance, thresholds, and explainability plots.
- `advanced_final_analysis/`: advanced-feature experiment metrics, probabilities, and SHAP summary.
- `final_error_analysis/`: error records, threshold-specific summaries, and plots.
- `final_shap_analysis/`: global, directional, missed/rescued-failure and per-case explainability artifacts.

Multiple scripts use different random splits, feature sets, or threshold protocols. Read the associated script and metadata before comparing tables; do not treat all files as one experiment or infer causality from SHAP attributions. Run scripts in `src/` to intentionally regenerate outputs. Some scripts overwrite existing files.
