import streamlit as st
import os

st.title("Legal RAG Assistant")

st.write("✅ Step 1: App Started")

st.write("Current Directory:")
st.write(os.getcwd())

st.write("Files in Current Directory:")
st.write(os.listdir("."))

st.write("✅ App loaded successfully")

question = st.text_input("Ask a question")

if question:
    st.write(f"You asked: {question}")
``
