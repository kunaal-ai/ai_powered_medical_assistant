"""
Evaluate fairness and bias of the trained model using AIF360.
Compares outcomes between privileged (sex=1) and unprivileged (sex=0) groups.
"""
import mlflow
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for environments without display

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# ------------------------------
# Step 1: Load data and model
# ------------------------------
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()
protected_attr = "sex"

# Ensure protected attribute is binary int (0 or 1)
X_test[protected_attr] = np.where(X_test[protected_attr] > 0.5, 1, 0).astype(int)

model = joblib.load("models/logisticregression_model.pkl")

# Drop protected attribute for prediction since model was trained without it
X_test_nosex = X_test.drop(columns=[protected_attr], errors='ignore')
y_pred = model.predict(X_test_nosex)

# ------------------------------
# Step 2: Prepare AIF360 datasets
# ------------------------------
# True labels dataset (only protected attribute + true label)
true_dataset = BinaryLabelDataset(
    df=pd.DataFrame({
        protected_attr: X_test[protected_attr],
        'label': y_test
    }),
    label_names=['label'],
    protected_attribute_names=[protected_attr],
    favorable_label=1,
    unfavorable_label=0
)

# Predicted labels dataset (only protected attribute + predicted label)
pred_dataset = BinaryLabelDataset(
    df=pd.DataFrame({
        protected_attr: X_test[protected_attr],
        'label': y_pred
    }),
    label_names=['label'],
    protected_attribute_names=[protected_attr],
    favorable_label=1,
    unfavorable_label=0
)

# ------------------------------
# Step 3: Compute fairness metrics
# ------------------------------
metric = ClassificationMetric(
    true_dataset,
    pred_dataset,
    privileged_groups=[{protected_attr: 1}],
    unprivileged_groups=[{protected_attr: 0}]
)

fairness_metrics = {
    "DisparateImpact": metric.disparate_impact(),
    "StatisticalParityDifference": metric.statistical_parity_difference(),
    "EqualOpportunityDifference": metric.equal_opportunity_difference(),
    "AverageOddsDifference": metric.average_odds_difference(),
    "TheilIndex": metric.theil_index()
}

# log to MLflow
mlflow.set_experiment("Diabetes Model Comparison")
with mlflow.start_run(run_name="Fairness_LogisticRegression"):
    mlflow.set_tag("type", "fairness")
    mlflow.set_tag("protected_attr", "sex")
    for key, value in fairness_metrics.items():
        mlflow.log_metric(key, value)
    mlflow.log_artifact("results/fairness_metrics.json")

# ------------------------------
# Step 4: Output results
# ------------------------------
print("\n✅ Fairness / Bias Metrics (Sex = protected attribute):")
for key, value in fairness_metrics.items():
    print(f"{key}: {value:.4f}")

# Save metrics to JSON file for later reference or reporting
with open("results/fairness_metrics.json", "w") as f:
    json.dump(fairness_metrics, f, indent=4)

print("\n✅ Fairness testing complete. Metrics saved to 'results/fairness_metrics.json'")
