import pandas as pd
import os
import joblib
import numpy as np
import pickle
import time
import json
import mlflow

from datetime import datetime
from dotenv import load_dotenv

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Suppress scikit-learn warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load environment variables from .env file
load_dotenv()

# ------------------------------
# Load scaler and model
# ------------------------------
model = joblib.load("models/logisticregression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
            'BMI', 'DiabetesPedigreeFunction', 'Age', 'sex']

# ------------------------------
# Step 1: Hardcoded input
# ------------------------------
input_data = {
    "Pregnancies": [2],
    "Glucose": [130],
    "BloodPressure": [80],
    "SkinThickness": [25],
    "Insulin": [100],
    "BMI": [40.5],
    "DiabetesPedigreeFunction": [0.7],
    "Age": [45],
    "sex": [1]
}
df = pd.DataFrame(input_data)

# ------------------------------
# Step 2: Input Validation
# ------------------------------
def validate_input(df):
    errors = []
    if df["Glucose"].values[0] <= 0:
        errors.append("Invalid Glucose value")
    if df["BMI"].values[0] <= 0:
        errors.append("Invalid BMI")
    if df["Age"].values[0] <= 0:
        errors.append("Invalid Age")
    if errors:
        raise ValueError(" ❌ Input validation failed: " + ", ".join(errors))

validate_input(df)

# ------------------------------
# Step 3: Predict Risk
# ------------------------------
input_df = pd.DataFrame(input_data, columns=features)
scaled = scaler.transform(input_df)
scaled_df = pd.DataFrame(scaled, columns=features)

# Predict using model trained without 'sex'
prediction = model.predict(scaled_df[features[:-1]])[0]
confidence = model.predict_proba(scaled_df[features[:-1]])[0][1]

# ------------------------------
# Step 4: Load VectorStore and LLM
# ------------------------------
with open("models/document_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding=embedding)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ------------------------------
# Step 5: Ask Question with Fallback
# ------------------------------
question = "What are common symptoms of diabetes?"
rag_answer = "⚠️ Unable to fetch answer at this time."

try:
    for _ in range(2):  # Retry once
        try:
            rag_result = qa_chain.invoke(question)
            rag_answer = rag_result['result']
            break
        except Exception as e:
            print("⚠️ Retry LLM due to error:", e)
            time.sleep(2)
except Exception as final_error:
    print("❌ LLM pipeline failed:", final_error)

# ------------------------------
# Step 6: Display Output
# ------------------------------
print("\n🩺 Diabetes Prediction:")
print(f"Prediction: {'Diabetic' if prediction else 'Not Diabetic'} (Confidence: {confidence:.2f})")

print("\n💬 Medical Assistant Answer:")
print(f"Q: {question}\nA: {rag_answer}")

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

log_entry = {
    "timestamp": datetime.utcnow().isoformat(),
    "input": input_data,
    "prediction": {
        "class": "Diabetic" if prediction else "Not Diabetic",
        "confidence": round(float(confidence), 4)
    },
    "qa": {
        "question": question,
        "answer": rag_answer
    }
}

with open("logs/app_log.jsonl", "a") as f:
    f.write(json.dumps(log_entry) + "\n")

print("📝 Log entry added.")

# Start MLflow run (can be inside or outside a loop)
mlflow.set_tracking_uri("http://localhost:5001")
with mlflow.start_run(run_name="FullAppInteraction"):

    # Log model prediction
    mlflow.log_metric("confidence", confidence)
    mlflow.log_param("predicted_class", "Diabetic" if prediction else "Not Diabetic")

    # Log full input as a dict
    mlflow.log_dict(input_data, "input_data.json")

    # Log QA query/response as text or JSON
    mlflow.log_dict({
        "question": question,
        "answer": rag_answer
    }, "qa_log.json")

    mlflow.set_tag("phase", "Phase 5")
    mlflow.set_tag("type", "prediction+rag")