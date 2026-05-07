# 📄 Agentic RAG System with LangGraph
An intelligent Document Question Answering System built using LangGraph, LangChain, FAISS, HuggingFace Embeddings, and Gemini 2.5 Flash.

This project allows users to upload PDF documents and ask natural language questions.
The system dynamically retrieves relevant chunks, evaluates evidence quality, and generates grounded answers using an agentic workflow.

## 🚀 Features
* 📂 Upload PDF documents
* 🧠 Semantic Retrieval using FAISS Vector Database
* ✂️ Intelligent Chunking Pipeline
* 🔍 Dynamic Similarity Threshold Retrieval
* 🤖 Agentic Workflow using LangGraph
* 📑 Evidence Evaluation (FULL, PARTIAL, INSUFFICIENT)
* 🧾 Context-Grounded Answer Generation
* ⚡ Streamlit UI
* 🐳 Docker Support
* 🧪 Jupyter Notebook Prototype Included

## 🏗️ Architecture
```
                ┌──────────────────┐
                │   Upload PDF     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ PDF Processing   │
                │ (Unstructured)   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Text Chunking    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Embeddings       │
                │ MiniLM-L6-v2     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ FAISS Vector DB  │
                └────────┬─────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │ LangGraph Agentic Pipeline  │
          └────────────┬────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼                              ▼
┌───────────────┐            ┌────────────────┐
│ Retrieve Docs │ ───────▶  │ Evidence Check │
└───────────────┘            └────────────────┘
                                       │
                                       ▼
                           ┌──────────────────┐
                           │ Generate Answer  │
                           └──────────────────┘
```
## 🧠 Agent Workflow

The system uses a 3-node LangGraph workflow:

**1️⃣ Retriever Node**\
Retrieves relevant chunks from FAISS
Uses dynamic thresholding based on similarity score

**2️⃣ Evidence Evaluator Node**\
Determines whether retrieved context is:
* FULL
* PARTIAL
* INSUFFICIENT

**3️⃣ Generator Node**\
Generates responses based on evidence quality:

* Fully grounded answers
* Hybrid contextual + general knowledge answers
* Graceful fallback responses

## 📁 Project Structure
```
.
├── main.py                 # Main Streamlit Application
├── ingestor.py             # PDF processing logic
├── chunking.py             # Chunking pipeline
├── vectorstore.py          # FAISS vector DB creation
├── warmup.py               # Model warmup script
├── requirements.txt
├── uploaded_files/
├── local_embedding_model/
├── attention.pdf
├── Working.ipynb           # Prototype notebook
└── Dockerfile
```
