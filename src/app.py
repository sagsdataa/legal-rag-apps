import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline
)

from transformers import pipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.write("Step 1: Starting app")

# Embedding model
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
st.write("Step 2: Loading embeddings")
embeddings = load_embeddings()

# Load vector DB
@st.cache_resource
def load_vector_db():
    return FAISS.load_local(
        "../faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
st.write("Step 3: Load Vector DB")
vectorDB = load_vector_db()

# Retriever
retriever = vectorDB.as_retriever(
    search_kwargs={"k": 3}
)


# LLM
@st.cache_resource
def load_llm():

    pipe = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=128
    )

    return HuggingFacePipeline(
        pipeline=pipe
    )

st.write("Step 4: Loading LLM")
llm = load_llm()

st.write("Step 5: Creating Pipeline")

llm = HuggingFacePipeline(
    pipeline=pipe
)

st.write("Step 6: App Loaded Successfully")

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
""")

# RAG Chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# UI
st.title("Legal RAG Assistant")

question = st.text_input(
    "Ask a question"
)

if question:
    answer = rag_chain.invoke(question)
    st.write(answer)
