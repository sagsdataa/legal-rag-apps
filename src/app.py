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


# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Load vector DB
vectorDB = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever
retriever = vectorDB.as_retriever(
    search_kwargs={"k": 3}
)

# LLM
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=128
)

llm = HuggingFacePipeline(
    pipeline=pipe
)

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
