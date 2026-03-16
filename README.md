# RAG AI Knowledge Assistant

This project is a Retrieval Augmented Generation (RAG) web app built with Streamlit.

The AI answers questions using custom documents stored in the data folder.

Features:
- Multi-file document retrieval
- AI answers grounded in context
- Source document verification
- Question history
- Clean Streamlit interface

Tech Stack:
- Python
- Streamlit
- LangChain
- FAISS Vector Database
- Sentence Transformers
- Ollama LLM

How to Run:

1. Install requirements

pip install -r requirements.txt

2. Run the app

streamlit run app_web.py

Then open:

http://localhost:8501
