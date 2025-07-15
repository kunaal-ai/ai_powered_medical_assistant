"""
Explainability with SHAP for Logistic Regression and Random Forest Models
Saves summary plots to results directory.
"""

import shap
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
import mlflow

mlflow.set_experiment("Diabetes Model Comparison")


warnings.filterwarnings("ignore")  # Ignore warnings for clean output

# Load the test dataset
X_test = pd.read_csv("data/X_test.csv")
X_test_nosex = X_test.drop(columns=["sex"], errors="ignore")

# Load trained models
lr_model = joblib.load("models/logisticregression_model.pkl")
rf_model = joblib.load("models/randomforest_model.pkl")

# ---------------------------------------
# Logistic Regression SHAP Explanation
# ---------------------------------------
print("🔍 Explaining Logistic Regression model...")

# Use KernelExplainer for linear models
explainer_lr = shap.KernelExplainer(lr_model.predict_proba, X_test_nosex[:100])
shap_values_lr = explainer_lr.shap_values(X_test_nosex[:100])

# SHAP summary plot path
summary_plot_path = "results/shap_summary_lr.png"
X_sample = X_test_nosex[:100]
plt.figure()
shap.summary_plot(shap_values_lr, X_sample, show=False)
plt.tight_layout()
plt.savefig(summary_plot_path, bbox_inches="tight")
plt.close()

# Start MLflow run
with mlflow.start_run(run_name="Explainability_LogisticRegression"):
    mlflow.set_tag("type", "explainability")
    
    # ✅ Use correct shape for SHAP values
    mean_abs_shap = np.abs(shap_values_lr[1]).mean(axis=0)

    for feature, val in zip(X_test_nosex.columns, mean_abs_shap):
        mlflow.log_metric(f"SHAP_{feature}", float(val))
    
    # Log plot
    mlflow.log_artifact(summary_plot_path)

# Fix for binary classification: shap_values returns list
if isinstance(shap_values_lr, list) and len(shap_values_lr) == 2:
    shap_values_to_plot = shap_values_lr[1]  # Class 1
else:
    shap_values_to_plot = shap_values_lr

shap.summary_plot(shap_values_to_plot, X_test_nosex[:100], show=False)
plt.title("SHAP Summary - Logistic Regression")
plt.savefig("results/shap_logistic_summary.png")
plt.close()


# ---------------------------------------
# Random Forest SHAP Explanation
# ---------------------------------------
print("🔍 Explaining Random Forest model...")

# Use TreeExplainer for tree-based models
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test_nosex)

# log average SHAP value
import mlflow
mlflow.set_experiment("Diabetes Model Comparison")

with mlflow.start_run(run_name="Explainability_RandomForest"):
    mean_abs_shap = np.abs(shap_values_lr[1]).mean(axis=0)
    for feature, val in zip(X_test_nosex.columns, mean_abs_shap):
        mlflow.log_metric(f"SHAP_{feature}", float(val))

shap.summary_plot(shap_values_rf, X_test_nosex, show=False)
plt.savefig("results/shap_randomforest_summary.png")
mlflow.log_artifact("results/shap_randomforest_summary.png")

# Check if it's a list (binary classification), then use class 1
if isinstance(shap_values_rf, list) and len(shap_values_rf) == 2:
    shap_values_to_plot_rf = shap_values_rf[1]  # Class 1
else:
    shap_values_to_plot_rf = shap_values_rf

# Save SHAP summary plot
shap.summary_plot(shap_values_to_plot_rf, X_test_nosex, show=False)
plt.title("SHAP Summary - Random Forest")
plt.savefig("results/shap_randomforest_summary.png")
plt.close()

print("\n✅ SHAP explainability complete. Visualizations saved in 'results/'")
