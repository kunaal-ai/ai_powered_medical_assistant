# AI-Powered Medical Assistant: Diabetes Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system for diabetes risk prediction with a focus on fairness and explainability.

## 🚀 Quick Start

1. **Setup**
   ```bash
   # Clone and enter project
   git clone https://github.com/yourusername/ai_powered_medical_assistant.git
   cd ai_powered_medical_assistant
   
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**
   ```bash
   # 1. Prepare data
   python scripts/01_data_preparation.py
   
   # 2. Train model
   python scripts/03_model_training.py

   # 2.2 Model Comparison
   python scripts/03_model_comparison.py
   
   # 3. Run analysis
   python scripts/04_bias_fairness_testing.py
   python scripts/05_model_explainability.py
   
   # 4. Run the application with MLflow tracking
   # In one terminal, start the MLflow server:
   mlflow server \
     --backend-store-uri ./mlruns \
     --default-artifact-root ./mlruns/artifacts \
     --host 0.0.0.0 \
     --port 5001
     
   # In another terminal, run the application
   python scripts/08_app_integeration_fallback.py
   ```

## 🔄 Load Testing with Locust

This project includes a [locustfile.py](locustfile.py) for load testing the FastAPI application.

### Quick Start
1. Install dependencies:
   ```bash
   pip install locust uvicorn fastapi
   ```

2. In one terminal, start the FastAPI app:
   ```bash
   uvicorn scripts.10_fastapi_app:app --reload --port 8000
   ```

3. In another terminal, run Locust:
   ```bash
   # Web UI mode
   locust --host=http://localhost:8000
   
   # Or headless mode (10 users, 1s spawn rate, 1 minute)
   locust --headless --users 10 --spawn-rate 1 --run-time 1m --host=http://localhost:8000
   ```

Access the web UI at http://localhost:8089 or check console for headless results. Modify `locustfile.py` to customize test scenarios.

## 🧠 Model Details

### Algorithm
- **Logistic Regression** with L2 regularization
- Balanced class weights for imbalanced data
- 5-fold cross-validation
- Key metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### Data Features
- **Medical Indicators**:
  - Glucose, Blood Pressure, BMI, Insulin
  - Age, Pregnancies, Diabetes Pedigree
  - Skin Thickness (synthetic: 0=female, 1=male)

### Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.85 |
| Precision | 0.83 |
| Recall | 0.82 |
| F1-Score | 0.82 |
| ROC-AUC | 0.91 |

## 📊 Model Monitoring & Analysis

### Performance Over Time
```
Accuracy Trend:
Week 1: ████████████████████ 85%
Week 2: ████████████████████▊ 86%
Week 3: ███████████████████▉ 84%
Week 4: ████████████████████ 85%
```

### Feature Distribution
```
Glucose Level Distribution:
<80     : ████ 50
80-100  : ████████████▊ 180
100-125 : ████████████████ 220
125-150 : ███████████ 150
150+    : ████▊ 80
```

### Data Quality Metrics
| Feature         | Missing % | Status  |
|-----------------|-----------|---------|
| Glucose         | 1.2%      | ✅ Good |
| BloodPressure   | 4.5%      | ⚠️ Fair |
| SkinThickness   | 29.6%     | ❌ Poor |
| Insulin         | 48.7%     | ❌ Poor |
| BMI             | 1.4%      | ✅ Good |
| Age             | 0.0%      | ✅ Good |

### Model Version History
| Version | Status  | Start Date | End Date   | Key Changes         |
|---------|---------|------------|------------|---------------------|
| v1.0.0  | ✅ Live | 2025-07-10 | 2025-07-15 | Initial deployment  |
| v1.1.0  | 🚧 Dev  | 2025-07-16 | 2025-07-20 | Feature engineering |

### Fairness Analysis
| Subgroup | Accuracy | FPR  | FNR  |
|----------|----------|------|------|
| Overall  | 0.85     | 0.12 | 0.18 |
| Female   | 0.84     | 0.11 | 0.20 |
| Male     | 0.86     | 0.13 | 0.15 |

### Performance by Age Group
```
20-30: ████████████████████ 88%
30-40: ██████████████████▊ 86%
40-50: ██████████████████ 84%
50-60: ████████████████▊ 82%
60+  : ████████████████ 80%
```

## 📊 MLflow Integration

Track experiments and model versions:

```bash
# Start MLflow UI
mlflow server --backend-store-uri sqlite:///mlruns.db
# View at http://localhost:5000

# Log custom metrics
with mlflow.start_run():
    mlflow.log_metric("roc_auc", 0.91)
    mlflow.log_param("model_type", "logistic_regression")
    mlflow.sklearn.log_model(model, "model")
```

### Monitoring Alerts
Set up alerts for:
- Model performance degradation (>5% drop in accuracy)
- Data drift detection
- Prediction distribution shifts
- Feature importance changes


