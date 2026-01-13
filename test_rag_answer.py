from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from src.rag import load_vectorstore, retrieve
from src.rag_answer import answer_from_docs

def main():
    # 1) Load embeddings (must match what you used for indexing)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2) Load existing Chroma store from disk
    vectordb = load_vectorstore(emb, persist_dir="chroma_store")

    # 3) Pick a RAG-style question
    question = "What are the AI initiatives mentioned by Microsoft?"

    # 4) Retrieve chunks
    docs = retrieve(vectordb, question, k=4)
    print(f"Retrieved: {len(docs)} docs")

    for i, d in enumerate(docs, 1):
        print(f"\n--- DOC {i} ---")
        print("SOURCE:", d.metadata.get("source"), "PAGE:", d.metadata.get("page"))
        print(d.page_content[:350])

    # 5) Generate grounded answer + citations
    answer, citations = answer_from_docs(question, docs)

    print("\n====================")
    print("FINAL ANSWER")
    print("====================")
    print(answer)

    print("\n====================")
    print("CITATIONS")
    print("====================")
    for c in citations:
        print(c)

if __name__ == "__main__":
    main()
