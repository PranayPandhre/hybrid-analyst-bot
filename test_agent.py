from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

from src.db import init_duckdb
from src.rag import load_vectorstore
from src.agent import answer


def main():
    # Init DB
    con = init_duckdb("data/financial_data.csv")

    # Init Vector DB
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = load_vectorstore(emb, persist_dir="chroma_store")

    # Conversation state (memory)
    state = {}

    questions = [
        "What is the market cap of Tesla?",
        "Compare Apple's revenue and Microsoft's revenue.",
        "What are the AI initiatives mentioned by Microsoft?",
        "What drove that growth?",  # follow-up should use memory/context
    ]

    for q in questions:
        result = answer(q, state, con, vectordb)

        print("\n====================")
        print("QUESTION:", q)
        print("====================")
        print("TRACE:", result.get("trace"))
        print("\nFINAL ANSWER:\n", result.get("final"))


if __name__ == "__main__":
    main()
