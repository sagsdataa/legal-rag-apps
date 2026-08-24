import streamlit as st
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


st.title("Legal RAG Assistant")

# Repository structure:
# legal-rag-apps/
# ├── faiss_index/
# │   ├── index.faiss
# │   └── index.pkl
# └── src/
#     └── app.py

REPO_ROOT = Path(__file__).resolve().parent.parent
FAISS_DIR = REPO_ROOT / "faiss_index"

st.write("FAISS location:", str(FAISS_DIR))
st.write("FAISS folder exists:", FAISS_DIR.exists())
st.write("index.faiss exists:", (FAISS_DIR / "index.faiss").exists())
st.write("index.pkl exists:", (FAISS_DIR / "index.pkl").exists())


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


st.write("Step 1: Loading embeddings")
embeddings = load_embeddings()
st.success("Step 2: Embeddings loaded")


@st.cache_resource
def load_vector_db():
    return FAISS.load_local(
        folder_path=str(FAISS_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


st.write("Step 3: Loading FAISS vector database")
vectorDB = load_vector_db()

st.success("Step 4: FAISS vector database loaded successfully")
st.write("Vectors stored:", vectorDB.index.ntotal)
