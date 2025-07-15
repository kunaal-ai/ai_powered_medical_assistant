"""
This script compares Logistic Regression and Random Forest models,
logs metrics and models using MLflow.
"""

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.models
from mlflow.models.signature import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Set MLflow experiment
mlflow.set_experiment("Diabetes Model Comparison")

# ----------------------------------------
# Load preprocessed training and test data
# ----------------------------------------
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Drop protected attribute 'sex' for fairness neutrality during training
X_train_nosex = X_train.drop("sex", axis=1)
X_test_nosex = X_test.drop("sex", axis=1)

# Define models to compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}



results = []

# ---------------------------
# Train and evaluate models
# ---------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\n🔍 Training {name}...")

        # Train
        model.fit(X_train_nosex, y_train)
        mlflow.log_params(model.get_params())


        # Predict
        y_pred = model.predict(X_test_nosex)
        y_proba = model.predict_proba(X_test_nosex)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        # Save results
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC AUC": roc
        })

        # Save model
        model_path = f"models/{name.lower()}_model.pkl"
        joblib.dump(model, model_path)

        # MLflow logging
        mlflow.log_param("model", name)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        })

        signature = infer_signature(X_test_nosex, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_test_nosex.iloc[:2])

        # Optional: log model file path as artifact
        mlflow.log_artifact(model_path)
       

# ----------------------------
# Save comparison summary
# ----------------------------
df_results = pd.DataFrame(results)
print("\n📊 Model Comparison:")
print(df_results.round(4))
df_results.to_csv("results/model_comparison.csv", index=False)

# --------------------------------------------
# Cross-validation for Random Forest (optional)
# --------------------------------------------
print("\n📊 Cross-Validation for Random Forest:")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_cv = cross_val_score(rf, X_train_nosex, y_train, cv=cv, scoring='roc_auc')
f1_cv = cross_val_score(rf, X_train_nosex, y_train, cv=cv, scoring='f1')

print(f"ROC AUC: {roc_auc_cv.mean():.4f} ± {roc_auc_cv.std():.4f}")
print(f"F1 Score: {f1_cv.mean():.4f} ± {f1_cv.std():.4f}")

cv_df = pd.DataFrame([
    {"Model": "RandomForest", "Metric": "ROC AUC", "Mean": roc_auc_cv.mean(), "Std": roc_auc_cv.std()},
    {"Model": "RandomForest", "Metric": "F1 Score", "Mean": f1_cv.mean(), "Std": f1_cv.std()},
])
cv_df.to_csv("results/cv_randomforest.csv", index=False)

print("\n✅ Model comparison complete. All logs and models saved.")
