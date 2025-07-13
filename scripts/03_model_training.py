'''
This script is used to train a machine learning model on structured data.

'''
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

# Load Preprocessed Data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()  # Flatten to 1D
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Initialize & Train Model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ---------------------------------------
# STEP: Cross-Validation for Training Set
# ---------------------------------------
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Initialize StratifiedKFold for cross-validation
# - n_splits=5: Split data into 5 folds
# - shuffle=True: Shuffle data before splitting for better distribution
# - random_state=42: Ensures reproducible results
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation for ROC AUC score
# - model: The logistic regression model to evaluate
# - scoring='roc_auc': Uses Area Under ROC Curve as the evaluation metric
# - cv=cv: Uses the StratifiedKFold configuration defined above
roc_auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

# Perform cross-validation for F1 score
# - scoring='f1': Uses F1 score (harmonic mean of precision and recall)
# - Useful for imbalanced datasets
f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

print("\n📊 Cross-Validation (Logistic Regression):")
print(f"ROC AUC: {roc_auc_scores.mean():.4f} ± {roc_auc_scores.std():.4f}")
print(f"F1 Score: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

cv_results =  {
    "metric": ["ROC AUC", "F1 Score"],
    "mean": [roc_auc_scores.mean(), f1_scores.mean()],
    "std": [roc_auc_scores.std(), f1_scores.std()]
}
pd.DataFrame(cv_results).to_csv("results/cv_logistic.csv", index = False)

# Predict on Test Data
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # for ROC-AUC

# Evaluate Model
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print(f"\n✅ Evaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc:.4f}")

# Save Model and Metrics
joblib.dump(model, "models/diabetes_model.pkl")

# Optional: Save metrics to file
metrics = {
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "roc_auc": roc
}
pd.DataFrame([metrics]).to_csv("results/model_metrics.csv", index=False)

print("\n Phase 2 Complete: Model and metrics saved.")
