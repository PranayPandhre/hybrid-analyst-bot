# src/rag.py
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def build_vectorstore(pdf_dir: str, embedding_fn, persist_dir: str = "chroma_store"):
    # Optional: map ticker -> company_name using your CSV
    ticker_to_name = {}
    csv_path = "data/financial_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        ticker_to_name = dict(zip(df["ticker"].astype(str), df["company_name"].astype(str)))

    all_docs = []
    for fn in os.listdir(pdf_dir):
        if not fn.lower().endswith(".pdf"):
            continue

        ticker = os.path.splitext(fn)[0].upper()  # "MSFT" from "MSFT.pdf"
        company_name = ticker_to_name.get(ticker, ticker)

        loader = PyPDFLoader(os.path.join(pdf_dir, fn))
        docs = loader.load()

        # add metadata for filtering later
        for d in docs:
            d.metadata["ticker"] = ticker
            d.metadata["company_name"] = company_name
            d.metadata["source"] = f"docs/{fn}"  # keep your existing style

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)

    vectordb = Chroma.from_documents(
        chunks,
        embedding_fn,
        persist_directory=persist_dir,
    )
    return vectordb

def load_vectorstore(embedding_fn, persist_dir: str = "chroma_store"):
    return Chroma(persist_directory=persist_dir, embedding_function=embedding_fn)



def retrieve(vectordb, query: str, k: int = 4, source_equals: str | None = None):
    """
    Retrieve docs from Chroma.
    If source_equals is set (e.g., 'docs/MSFT.pdf'), we constrain retrieval to that file.

    Why:
    - Prevent cross-company leakage (Meta text answering Microsoft question).
    - Makes citations accurate and reduces hallucination.
    """
    if source_equals:
        # Try Chroma metadata filtering first
        try:
            docs = vectordb.similarity_search(
                query,
                k=k,
                filter={"source": source_equals},
            )
            return dedup_docs(docs)
        except Exception:
            # Fallback: retrieve more, then filter in Python
            docs = vectordb.similarity_search(query, k=max(k * 5, 20))
            docs = [d for d in docs if d.metadata.get("source") == source_equals]
            return dedup_docs(docs)[:k]

    docs = vectordb.similarity_search(query, k=k)
    return dedup_docs(docs)


import re
from collections import Counter

def dedup_docs(docs):
    seen = set()
    out = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:120])
        if key not in seen:
            out.append(d)
            seen.add(key)
    return out

def infer_tickers_from_question(question: str, known_tickers: list[str]) -> list[str]:
    q = question.upper()
    hits = [t for t in known_tickers if re.search(rf"\b{re.escape(t)}\b", q)]
    return hits

def infer_ticker_from_retrieved_docs(docs) -> str | None:
    tickers = [d.metadata.get("ticker") for d in docs if d.metadata.get("ticker")]
    if not tickers:
        return None
    return Counter(tickers).most_common(1)[0][0]

def retrieve_semantic_company(vectordb, query: str, k: int = 4, global_k: int = 20):
    """
    Semantic retrieval that auto-detects company intent.
    1) retrieve global_k across all docs
    2) infer the most likely ticker
    3) retrieve again filtered to that ticker
    """
    # Stage A: global retrieval
    global_docs = vectordb.similarity_search(query, k=global_k)
    global_docs = dedup_docs(global_docs)

    # Try infer from question if possible
    # known tickers can be inferred from what exists in vectordb by looking at sources
    known_tickers = sorted({d.metadata.get("ticker") for d in global_docs if d.metadata.get("ticker")})
    tickers_in_q = infer_tickers_from_question(query, known_tickers)
    if tickers_in_q:
        target_ticker = tickers_in_q[0]
    else:
        # fallback: infer from which ticker dominates top retrieved docs
        target_ticker = infer_ticker_from_retrieved_docs(global_docs)

    if not target_ticker:
        # No company inferred → return best global docs
        return global_docs[:k], {"mode": "global", "target_ticker": None}

    # Stage C: company-filtered retrieval
    try:
        filtered = vectordb.similarity_search(
            query, k=max(k * 2, 8),
            filter={"ticker": target_ticker},
        )
        filtered = dedup_docs(filtered)[:k]
        return filtered, {"mode": "filtered", "target_ticker": target_ticker}
    except Exception:
        # Fallback: filter in Python if metadata filter unsupported
        filtered = [d for d in global_docs if d.metadata.get("ticker") == target_ticker]
        return filtered[:k], {"mode": "filtered_fallback", "target_ticker": target_ticker}


