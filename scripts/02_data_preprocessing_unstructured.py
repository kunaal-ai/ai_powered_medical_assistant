'''
This script is used to preprocess unstructured data into a format that can be used for training a machine learning model.
It takes a JSON file as input and converts it into a list of documents, where each document is a question and answer pair.
The script then splits the documents into chunks of a specified size and saves them to a pickle file.
The script also saves the original Q&A file for reference.

'''
import json
import pickle
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# TODO: replace this with a real JSON file like MedQuAD or HealthTap
faq_data = [
    {
        "question": "What are common symptoms of diabetes?",
        "answer": "Frequent urination, excessive thirst, fatigue, and blurred vision are common symptoms."
    },
    {
        "question": "Can type 2 diabetes be reversed?",
        "answer": "In some cases, lifestyle changes like diet and exercise can help manage or reverse type 2 diabetes."
    },
    {
        "question": "How is diabetes diagnosed?",
        "answer": "Diabetes is diagnosed using tests such as fasting blood glucose, HbA1c, and oral glucose tolerance test."
    }
]

# Convert Q&A to LangChain Documents
documents = [
    Document(page_content=f"Q: {item['question']} A: {item['answer']}")
    for item in faq_data
]

# Chunk Documents for RAG
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

print(f"\n Document Chunks Created: {len(chunks)}")
print("Sample Chunk Preview:\n", chunks[0].page_content)

# Save Document Chunks for Later Use
with open("models/document_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

# Save original Q&A file for reference
with open("medical_faq.json", "w") as f:
    json.dump(faq_data, f, indent=2)

print("\n Unstructured data preparation complete. Chunks saved for Phase 5.")
