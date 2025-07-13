# PHASE 2 EXTENDED: Compare Logistic Regression vs Random Forest
'''
This script is used to compare the performance of two machine learning models on structured data.

It takes two models (Logistic Regression and Random Forest) and compares their performance on a test set.

The script also performs cross-validation for the Random Forest model to evaluate its performance on the training set.

The script saves the models and results to disk for later use.
'''
from operator import index
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Load preprocessed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

# Train, predict, and evaluate both models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba)
    }

    results.append(metrics)
    joblib.dump(model, f"models/{name.lower()}_model.pkl")

# Show comparison
df_results = pd.DataFrame(results)
print("\n📊 Model Comparison:")
print(df_results.round(4))

# Save results to CSV
df_results.to_csv("results/model_comparison.csv", index=False)

# ---------------------------------------
# Cross-Validation for Training Set
# ---------------------------------------
from sklearn.model_selection import cross_val_score, StratifiedKFold
output_path = "results/cross_validation_scores.csv"
# Initialize new RandomForest instance for cross-validation
rf = RandomForestClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='roc_auc')
f1_rf = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1')

print("\n📊 Cross-Validation Results for Random Forest:")
print(f"ROC AUC: {roc_auc_rf.mean():.4f} ± {roc_auc_rf.std():.4f}")
print(f"F1 Score: {f1_rf.mean():.4f} ± {f1_rf.std():.4f}")

cv_results = pd.DataFrame([
    {"Model":"RandomForest", "Metric":"ROC AUC", "Mean": roc_auc_rf.mean(), "Std": roc_auc_rf.std()},
    {"Model":"RandomForest", "Metric":"F1 Score", "Mean": f1_rf.mean(), "Std": f1_rf.std()},
])
cv_results.to_csv("results/cv_randomforest.csv", index=False)

print("\n Model comparison complete. Models and results saved.")
