# NLP Final Project 2026 - RAG Chatbot
**Master in Machine Learning for Health**

## Project Overview
This application implements a Retrieval-Augmented Generation (RAG) system to answer questions from a document database while mitigating hallucinations.

## Functional Specifications Met
- **Vector Database**: Implemented using FAISS.
- **Open-Source LLMs**: Powered by Llama 3.1 8B via UC3M API.
- **Multi-language**: Supports English and Spanish queries.
- **Citations**: Responses include source document references.

## Installation
1. Install dependencies: `pip install -r requirements.txt`
2. Set `UC3M_API_KEY` in your terminal → `$env:UC3M_API_KEY="api key here"`
3. Run the app: `streamlit run app.py`

## Design Decisions 
- **Chunking Strategy**: Documents are halved to ensure precise retrieval.
- **Top-K Selection**: Defaulting to k=2 to balance context breadth and LLM token limits.
