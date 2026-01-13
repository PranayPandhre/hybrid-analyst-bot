from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from src.rag import build_vectorstore

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = build_vectorstore(
    pdf_dir="docs",
    embedding_fn=emb,
    persist_dir="chroma_store"
)

# Print chunk count so you know indexing worked
try:
    count = vectordb._collection.count()
    print("RAG index built:", count, "chunks")
except Exception:
    print("RAG index built (count unavailable, but build succeeded).")
