import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.title("Legal RAG Assistant")

st.write("Step 1: Start")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

st.write("✅ Step 2: Embeddings Loaded")

@st.cache_resource
def load_vector_db():
    return FAISS.load_local(
        "../faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorDB = load_vector_db()

st.write("✅ Step 3: FAISS Loaded")
