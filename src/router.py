# src/router.py
from __future__ import annotations

import json
from typing import Dict, Any

from src.llm import chat_completion

ROUTES = {"SQL", "RAG"}

_ROUTER_SYSTEM = (
    "You are a routing function for a hybrid SQL + RAG assistant.\n"
    "Return ONLY valid JSON. No markdown. No extra keys.\n"
    "You must choose one route: SQL, RAG, or BOTH.\n"
    "\n"
    "Use SQL when:\n"
    "- The user asks for numeric facts present in the table (market cap, revenue, net income, EPS, etc.)\n"
    "- The user asks to compare companies using table columns\n"
    "\n"
    "Use RAG when:\n"
    "- The user asks qualitative questions from PDFs (strategy, risks, initiatives, commentary)\n"
    "- The user asks 'why'/'how' based on transcript content\n"
    "\n"
    "Use BOTH when:\n"
    "- The question needs table numbers plus narrative explanation from PDFs\n"
)

_ROUTER_USER_TEMPLATE = """Decide the route for this user question.

Question: {question}

Output schema (must match exactly):
{{
  "route": "SQL" | "RAG" ,
  "reason": "one short sentence"
}}
"""


def _safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON even if model accidentally adds extra text.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def route_query(question: str) -> Dict[str, str]:
    raw = chat_completion(
        [
            {"role": "system", "content": _ROUTER_SYSTEM},
            {"role": "user", "content": _ROUTER_USER_TEMPLATE.format(question=question)},
        ],
        temperature=0.0,
    )

    data = _safe_json_extract(raw)
    route = data.get("route")
    reason = data.get("reason", "")

    if route not in ROUTES:
        # Hard fallback if it returns something weird
        return {"route": "RAG", "reason": "Invalid route returned; defaulting to RAG."}

    return {"route": route, "reason": reason}
