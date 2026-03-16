# 🤖 AI Knowledge Assistant (RAG Powered)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![AI](https://img.shields.io/badge/AI-RAG-green)

An AI-powered knowledge assistant that answers questions using **custom documents**.
Built using **Retrieval-Augmented Generation (RAG)** so responses are grounded in real context instead of hallucinating.

---

# 🚀 Demo

Ask questions about AI concepts like:

* What is Machine Learning?
* Difference between Machine Learning and Deep Learning.
* What are the applications of AI?

The assistant retrieves relevant context from documents and generates an answer.

---

# 🖼 App Preview

![image alt](https://github.com/Sameekshavermaa/RAG-AI-assistant/blob/bdf88c62671a57188f05ec68b476d909ee361b43/Screenshot%202026-03-16%20at%2001.20.30.png)

![image alt](https://github.com/Sameekshavermaa/RAG-AI-assistant/blob/bdf88c62671a57188f05ec68b476d909ee361b43/Screenshot%202026-03-16%20at%2001.20.30.png)

![image alt](https://github.com/Sameekshavermaa/RAG-AI-assistant/blob/bdf88c62671a57188f05ec68b476d909ee361b43/Screenshot%202026-03-16%20at%2001.20.30.png)

---

# ✨ Features

✔ AI answers based on your documents
✔ Multi-file document retrieval
✔ Source file references
✔ Clean Streamlit interface
✔ Vector search using FAISS

---

# 🧠 How It Works

1️⃣ Documents are stored in the **data folder**

2️⃣ Text is converted into **embeddings using Sentence Transformers**

3️⃣ Embeddings are stored in **FAISS vector database**

4️⃣ When a user asks a question:

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

# 📂 Project Structure

rag-ai-assistant
│
├── app_web.py
├── requirements.txt
├── README.md
│
└── data
    ├── ai.txt
    ├── machine_learning.txt
    ├── deep_learning.txt
    ├── nlp.txt
    └── computer_vision.txt

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

# 📌 Future Improvements

* ChatGPT-style conversation UI
* Upload your own documents
* Export answers as PDF
* Deploy online

---

# 👩‍💻 Author

Sameeksha Verma
