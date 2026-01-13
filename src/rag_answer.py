# src/rag_answer.py
from __future__ import annotations

from typing import List, Dict, Tuple
from src.llm import chat_completion
import json


def answer_from_docs(question: str, docs) -> Tuple[str, List[Dict]]:
    context_blocks = []
    citations: List[Dict] = []
    tickers_in_docs = []

    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        ticker = d.metadata.get("ticker", "unknown")

        tickers_in_docs.append(ticker)
        context_blocks.append(f"[{i}] (ticker={ticker}, source={src}, page={page})\n{d.page_content}")
        citations.append({"chunk": i, "source": src, "page": page})

    # unique tickers in the same order they appeared
    seen = set()
    unique_tickers = []
    for t in tickers_in_docs:
        if t not in seen:
            unique_tickers.append(t)
            seen.add(t)

    context = "\n\n".join(context_blocks)

    system = (
        "You are a careful analyst.\n"
        "Return ONLY valid JSON. No markdown.\n"
        "You MUST follow rules:\n"
        "1) Use ONLY the provided chunks.\n"
        "2) Create an output section for EVERY ticker listed in `tickers`.\n"
        "3) Under each ticker, include ONLY bullets supported by chunks with that SAME ticker.\n"
        "4) Each bullet must include citation chunk ids and a short evidence quote (5–15 words).\n"
        "5) If there is no relevant evidence for a ticker, return an empty bullets list.\n"
    )

    user = (
        f"Question: {question}\n\n"
        f"tickers: {unique_tickers}\n\n"
        "Chunks:\n"
        f"{context}\n\n"
        "Return JSON in this exact schema:\n"
        "{\n"
        '  "sections": [\n'
        '    {\n'
        '      "ticker": "MSFT",\n'
        '      "source": "docs/MSFT.pdf",\n'
        '      "bullets": [\n'
        '        {\n'
        '          "text": "....",\n'
        '          "cites": [2],\n'
        '          "evidence": "quoted words"\n'
        '        }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        "}\n"
    )

    raw = chat_completion(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        temperature=0.0,
    )

    # Parse JSON safely (basic cleanup if model adds stray text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: try to extract the first {...} block
        start = raw.find("{")
        end = raw.rfind("}")
        data = json.loads(raw[start:end+1])

    # Render into the text format you want in the UI
    lines = []
    for sec in data.get("sections", []):
        ticker = sec.get("ticker", "unknown")
        source = sec.get("source", "unknown")
        bullets = sec.get("bullets", [])

        lines.append(f"From {ticker} ({source}):")
        if not bullets:
            lines.append("No relevant evidence in provided chunks.")
        else:
            for b in bullets:
                cites = "".join([f"[{c}]" for c in b.get("cites", [])])
                evidence = b.get("evidence", "")
                text = b.get("text", "").strip()
                lines.append(f"• {text} {cites} (evidence: \"{evidence}\")")
        lines.append("")  # blank line

    answer = "\n".join(lines).strip()
    return answer, citations

