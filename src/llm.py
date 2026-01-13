# import os
# from openai import OpenAI

# def get_client():
#     # For Groq, you typically set base_url + api_key
#     return OpenAI(
#         api_key=os.environ["GROQ_API_KEY"],
#         base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
#     )

# def chat_completion(messages, model="llama-3.1-8b-instant", temperature=0):
#     client = get_client()
#     resp = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#     )
#     return resp.choices[0].message.content


# src/llm.py
from __future__ import annotations

import os
from typing import List, Dict, Optional

from openai import OpenAI


def _get_env(name: str, default: Optional[str] = None) -> str:
    """Get environment variable or raise a helpful error."""
    val = os.environ.get(name, default)
    if val is None or val.strip() == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Fix: add it to your .env (recommended) or export it in your shell."
        )
    return val


def get_client() -> OpenAI:
    """
    Returns an OpenAI-compatible client configured for Groq.

    Required:
      - GROQ_API_KEY

    Optional:
      - GROQ_BASE_URL (defaults to Groq OpenAI-compatible endpoint)
    """
    api_key = _get_env("GROQ_API_KEY")
    base_url = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    return OpenAI(api_key=api_key, base_url=base_url)


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> str:
    """
    Simple chat wrapper:
      messages = [{"role":"user","content":"..."}, ...]
    """
    client = get_client()

    # Allow model override via env var or function arg
    model_name = model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

