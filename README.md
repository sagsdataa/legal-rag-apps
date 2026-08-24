# Legal Document RAG Assistant

## Overview
This project implements a Retrieval-Augmented Generation (RAG) solution for legal documents using LangChain, Hugging Face, FAISS, and Streamlit.

The system processes legal documents, generates embeddings, stores them in a FAISS vector database, retrieves relevant document chunks, and uses a Large Language Model (LLM) to generate context-aware answers.

## Features
- Document loading and preprocessing
- Text chunking and embedding generation
- FAISS vector database for semantic search
- Retrieval-Augmented Generation (RAG)
- Question answering over legal documents
- Streamlit-based web interface

## Technology Stack
- Python
- LangChain
- Hugging Face Transformers
- FAISS
- Streamlit
- Sentence Transformers

## Project Workflow
Documents
→ Preprocessing
→ Chunking
→ Embeddings
→ FAISS Vector Database
→ Retriever
→ LLM
→ Answer Generation

## Installation

```bash
pip install -r requirements.txt
``
