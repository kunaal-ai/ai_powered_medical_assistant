import pandas as pd
import os
import joblib
import numpy as np
import pickle
import logging
from dotenv import load_dotenv

load_dotenv()
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Setup logging
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_resources():
    """Load model, scaler, and QA components."""
    try:
        model = joblib.load("models/logisticregression_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        with open("models/document_chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        embedding = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embedding=embedding)
        retriever = vectorstore.as_retriever()
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        logging.info("Resources loaded successfully.")
        return model, scaler, qa_chain
    except Exception as e:
        logging.error(f"Error loading resources: {e}")
        raise

def predict_diabetes(input_dict, model, scaler):
    """Scale input and predict diabetes risk."""
    try:
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age', 'sex']

        # Prepare input DataFrame
        input_df = pd.DataFrame(input_dict, columns=features)

        # Scale input
        scaled_features = scaler.transform(input_df)
        input_df_scaled = pd.DataFrame(scaled_features, columns=features)

        # Model trained without 'sex' column
        X_input = input_df_scaled.drop(columns=['sex'])

        prediction = model.predict(X_input)[0]
        confidence = model.predict_proba(X_input)[0][1]
        logging.info(f"Prediction made: {prediction} with confidence {confidence:.4f}")
        return prediction, confidence
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

def ask_question(question, qa_chain):
    """Query the QA chain and return an answer."""
    try:
        response = qa_chain.invoke(question)
        answer = response.get('result', "No answer generated")
        logging.info(f"Question asked: {question}")
        logging.info(f"Answer received: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Error during QA: {e}")
        raise

if __name__ == "__main__":
    try:
        model, scaler, qa_chain = load_resources()

        # Hardcoded sample input
        input_data = {
            "Pregnancies": [2],
            "Glucose": [130],
            "BloodPressure": [80],
            "SkinThickness": [25],
            "Insulin": [100],
            "BMI": [30.5],
            "DiabetesPedigreeFunction": [0.7],
            "Age": [45],
            "sex": [1]
        }
        question = "What are common symptoms of diabetes?"

        prediction, confidence = predict_diabetes(input_data, model, scaler)
        rag_answer = ask_question(question, qa_chain)

        print("\n Diabetes Prediction:")
        print(f"Prediction: {'Diabetic' if prediction else 'Not Diabetic'} (Confidence: {confidence:.2f})")

        print("\n Medical Assistant Answer:")
        print(f"Q: {question}\nA: {rag_answer}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
