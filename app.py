from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# 1️⃣ Load document
import os

# Load all .txt files in the data folder
documents = []
data_folder = "data/"
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join(data_folder, filename))
        docs = loader.load()
        documents.extend(docs)

# 2️⃣ Split document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 3️⃣ Convert chunks to text
texts = [chunk.page_content for chunk in chunks]

# 4️⃣ Create embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

# 5️⃣ Create FAISS vector database
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Vector database created with", index.ntotal, "chunks")

# 6️⃣ Ask questions with LLM
while True:
    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        break

    # 6a — get embedding of question
    question_embedding = model.encode([question])

    # 6b — search top 3 similar chunks
    D, I = index.search(np.array(question_embedding), k=3)

    # 6c — combine context
    context = "\n".join([texts[i] for i in I[0]])

    # 6d — prompt for LLM
    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    # 6e — call LLM
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nAI Answer:\n")
    print(response["message"]["content"])