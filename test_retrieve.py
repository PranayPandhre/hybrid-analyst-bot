from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from src.rag import load_vectorstore, retrieve_semantic_company

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = load_vectorstore(emb, persist_dir="chroma_store")

query = "What are the AI initiatives mentioned by Microsoft?"
docs, info = retrieve_semantic_company(db, query, k=3, global_k=25)
print("INFO:", info)

print("Retrieved:", len(docs), "docs")
for i, d in enumerate(docs, 1):
    print("\n--- DOC", i, "---")
    print("SOURCE:", d.metadata.get("source"), "PAGE:", d.metadata.get("page"))
    print(d.page_content[:350])
