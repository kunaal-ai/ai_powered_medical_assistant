import pickle
import os
import pytest
import warnings
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


load_dotenv()

@pytest.fixture(scope="module")
def qa_chain():
    with open("models/document_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding=embedding)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def test_rag_response_not_empty(qa_chain):
    result = qa_chain.invoke("What is diabetes?")
    assert "diabetes" in result['result'].lower() or len(result['result']) > 10, "Empty or irrelevant response"

def test_rag_handles_nonsense_query(qa_chain):
    result = qa_chain.invoke("flibbertygibbet xylophone moonwalk?")
    assert isinstance(result['result'], str), "Output should be a string"
