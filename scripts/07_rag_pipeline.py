import os
import json
import pickle
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Safety and hallucination detection
def is_safe_prompt(prompt: str) -> bool:
    blocked = ["suicide", "kill", "harm", "overdose", "violence"]
    prompt_lower = prompt.lower()
    for word in blocked:
        if word in prompt_lower:
            return False
    return True

def is_hallucinated(query: str, result: dict, retriever) -> bool:
    answer = result.get("result", "")
    if not answer:
        return False
    
    docs = retriever.invoke(query)
    top_context = " ".join(doc.page_content for doc in docs[:3])
    return answer.lower() not in top_context.lower()

load_dotenv()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Create FAISS Index
print("🔄 Creating FAISS index...")

with open("models/document_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"✅ Loaded {len(chunks)} document chunks")
vector_index = FAISS.from_documents(chunks, embedding_model)

# Save the index
os.makedirs("vectorstores", exist_ok=True)
vector_index.save_local("vectorstores/medical_faq_index")
print("✅ Saved FAISS index")

retriever = vector_index.as_retriever(search_kwargs={"k": 3})
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
print("✅ QA pipeline initialized.")

# --------------------------------------------
# 📊 RAGAS Evaluation Function
# --------------------------------------------
def run_ragas_evaluation():
    from datasets import Dataset

    eval_samples = [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Frequent urination, excessive thirst, fatigue, and blurred vision.",
            "context": "Q: What are common symptoms of diabetes? A: Frequent urination, excessive thirst, fatigue, and blurred vision."
        }
    ]

    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in eval_samples:
        result = qa_chain.invoke({"query": sample["question"]})
        docs = retriever.invoke(sample["question"])
        
        questions.append(sample["question"])
        answers.append(result["result"])
        contexts.append([" ".join(doc.page_content for doc in docs[:3])])
        ground_truths.append([sample["answer"]])

    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    })

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
        raise_exceptions=False
    )

    os.makedirs("evaluation", exist_ok=True)
    df = result.to_pandas()
    df.to_csv("evaluation/ragas_results.csv", index=False)

    print("\n📊 Evaluation Metrics:")
    print(f"Faithfulness: {df['faithfulness'].iloc[0]:.4f}")
    print(f"Answer Relevancy: {df['answer_relevancy'].iloc[0]:.4f}")
    print("✅ Results saved to /evaluation/")

# Interactive loop
print("\n🤖 Medical QA System")
print("Enter 'eval' to evaluate | 'exit' to quit\n")

while True:
    query = input("Your Question: ").strip()
    
    if query.lower() == "exit":
        break
    if query.lower() == "eval":
        run_ragas_evaluation()
        continue
        
    if not is_safe_prompt(query):
        print("⚠️ Unsafe or harmful query detected.")
        continue

    result = qa_chain.invoke({"query": query})
    print(f"\n💬 {result['result']}")
    
    if is_hallucinated(query, result, retriever):
        print("⚠️ Warning: Potential hallucination detected.")
    print()
