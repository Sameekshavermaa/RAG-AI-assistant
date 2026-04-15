import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

data_folder = os.path.join(os.path.dirname(__file__), "Data")

if not os.path.exists(data_folder):
    st.error("Data folder not found. Please check your repository structure.")
    st.stop()

for filename in os.listdir(data_folder):
    ...
# ------------------------
# Streamlit page config
# ------------------------
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    layout="wide",
    page_icon="🤖"
)
st.markdown("""
<div style='background-color:#4B0082;padding:15px;border-radius:10px'>
    <h1 style='color:white;text-align:center'>📚 RAG Knowledge Assistant 📚</h1>
    <p style='color:white;text-align:center'>Ask questions and get answers from your documents!</p>
</div>
""", unsafe_allow_html=True)
# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("⚙️ Settings")
question_input = st.sidebar.text_input("Ask a question:")

# ------------------------
# Load documents
# ------------------------
st.title("🧠 AI Knowledge Hub")
with st.spinner("📦 Loading documents and building vector database..."):
    documents = []
    data_folder = "Data"  # capital D — must match the repo folder exactly
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_folder, filename))
            docs = loader.load()
            documents.extend(docs)

    # Split docs into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

st.success("✅ Vector DB ready!")
# ------------------------
# Embeddings & FAISS index
# ------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

st.sidebar.success(f"Vector DB: {index.ntotal} chunks loaded ✅")

# ------------------------
# Question-answer logic
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if question_input:
    # Embed question
    question_embedding = model.encode([question_input])

    # Search top 3 chunks
    D, I = index.search(np.array(question_embedding), k=3)

    # Combine context
    context = "\n".join([texts[i] for i in I[0]])

    # Grounded prompt
    prompt = f"""
You are an AI assistant. Answer the question ONLY using the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question_input}

Answer:
"""

    # Call LLM
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

response = client.chat.completions.create(
    model="llama3-8b-8192",   # same Llama 3, hosted by Groq
    messages=[{"role": "user", "content": prompt}]
)
answer = response.choices[0].message.content

    # Save in history
    st.session_state.history.append({
        "question": question_input,
        "answer": answer_text,
        "chunks": [chunks[i] for i in I[0]]
    })

# ------------------------
# Display Q&A history
# ------------------------
for qa in reversed(st.session_state.history):
    with st.expander(f"📝 Question: {qa['question']}"):
        st.markdown("### 🤖 AI Answer")
        st.info(qa["answer"])

        st.markdown("### 📄 Context Chunks Used (with source file)")
        for i, chunk in enumerate(qa["chunks"], start=1):
            st.write(f"**Chunk #{i} | Source:** {chunk.metadata['source']}")
            st.write(chunk.page_content)
            st.write("---")
