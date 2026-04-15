import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import os

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

data_folder = os.path.join(os.path.dirname(__file__), "Data")

if not os.path.exists(data_folder):
    st.error("Data folder not found. Please check your repository structure.")
    st.stop()

with st.spinner("📦 Loading documents and building vector database..."):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_folder, filename))
            docs = loader.load()
            documents.extend(docs)

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
    question_embedding = model.encode([question_input])
    D, I = index.search(np.array(question_embedding), k=3)
    
    # FIX 1: Truncate context to 2000 chars
    context = "\n".join([texts[i] for i in I[0]])
    context = context[:2000]

    # FIX 3: Tighter prompt
    prompt = f"Answer ONLY using this context. If unsure, say 'I don't know'.\n\nContext:\n{context}\n\nQuestion: {question_input}\n\nAnswer:"

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content

    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")  # shows the real error
        st.stop()
    st.session_state.history.append({
        "question": question_input,
        "answer": answer,
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
