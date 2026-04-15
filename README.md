# 🤖 AI Knowledge Assistant (RAG Powered)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

An AI-powered knowledge assistant that answers questions using **custom documents**.
Built using **Retrieval-Augmented Generation (RAG)** so responses are grounded in real context instead of hallucinating.

---

# 🚀 Demo

Link: https://rag-ai-assistant-jqoukck5ljt9lsch6qpcpq.streamlit.app/

Ask questions about AI concepts like:

* What is Machine Learning?
* Difference between Machine Learning and Deep Learning.
* What are the applications of AI?

The assistant retrieves relevant context from documents and generates an answer.

---

# 🖼 App Preview

![image alt](https://github.com/Sameekshavermaa/RAG-AI-assistant/blob/bdf88c62671a57188f05ec68b476d909ee361b43/Screenshot%202026-03-16%20at%2001.20.30.png)

![image alt](https://github.com/Sameekshavermaa/RAG-AI-assistant/blob/8016ae2e34796a17956c119420228f413df6c11f/Screenshot%202026-03-16%20at%2020.50.10.png)

![image alt](https://github.com/Sameekshavermaa/RAG-AI-assistant/blob/8016ae2e34796a17956c119420228f413df6c11f/Screenshot%202026-03-16%20at%2020.35.43.png)

---
## ✨ Key Features

- 🔎 Document-based question answering
- 🧠 Retrieval-Augmented Generation (RAG)
- 📚 Multi-document knowledge base
- ⚡ Fast semantic search using FAISS
- 🎨 Clean Streamlit UI
- 📂 Source references for transparency

---
## 🧠 How It Works

User Question  
↓  
Convert the question to an embedding  
↓  
Search similar chunks in the FAISS vector database  
↓  
Retrieve relevant document context  
↓  
Send context + question to LLM  
↓  
Generate grounded AI response

* The system retrieves relevant text chunks
* Sends them to the LLM
* Generates a grounded response

This technique is called **Retrieval-Augmented Generation (RAG)**.

---

# 🛠 Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Sentence Transformers
* Ollama LLM

---

## 📂 Project Structure

```
rag-ai-assistant/
│
├── app_web.py
├── requirements.txt
├── README.md
│
└── data/
    ├── ai.txt
    ├── machine_learning.txt
    ├── deep_learning.txt
    ├── nlp.txt
    └── computer_vision.txt
```
---

# ⚙ Installation

Clone the repository

git clone https://github.com/sameekshavermaa/rag-ai-assistant.git

Install dependencies

pip install -r requirements.txt

Run the app

streamlit run app_web.py

Open browser

http://localhost:8501

---
## 💬 Example Query

Question:
What is the difference between Machine Learning and Deep Learning?

AI Response:
Machine Learning is a subset of AI where systems learn from data.
Deep Learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns.

---
# 📌 Future Improvements

* ChatGPT-style conversation UI
* Upload your own documents
* Export answers as PDF
* Deploy online

---

# 👩‍💻 Author

Sameeksha Verma
