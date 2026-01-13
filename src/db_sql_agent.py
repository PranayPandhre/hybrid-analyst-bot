# src/db_sql_agent.py
from __future__ import annotations

import json
from typing import Any, Dict

from src.llm import chat_completion
from src.schemas import FIN_SCHEMA
from src.db import TABLE_NAME


SQL_PROMPT = """
You are a SQL generator for DuckDB.

Return ONLY valid JSON. No markdown. No explanation.
Schema:
{schema}

Task:
Generate a single SQL SELECT query over the table "{table}" that answers the question.

Rules:
- Output must be JSON exactly like: {{"sql": "SELECT ..."}}
- Only SELECT queries are allowed.
- Do not use INSERT/UPDATE/DELETE/DROP/ALTER/CREATE.
- Always query from "{table}".
- Use correct column names from the schema.

Question:
{q}
"""

REPAIR_PROMPT = """
You are a SQL repair function for DuckDB.

Return ONLY valid JSON. No markdown. No explanation.
Schema:
{schema}

We tried this SQL (it failed):
{bad_sql}

Error:
{err}

Task:
Return a corrected SQL SELECT query that answers the question.

Rules:
- Output must be JSON exactly like: {{"sql": "SELECT ..."}}
- Only SELECT queries are allowed.
- Always query from "{table}".
- Use correct column names from the schema.

Question:
{q}
"""


def _safe_json_extract(text: str) -> Dict[str, Any]:
    """
    Parse JSON even if the model accidentally includes extra text.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def is_safe_sql(sql: str) -> bool:
    s = sql.strip().lower()

    if not s.startswith("select"):
        return False
    bad = ["insert", "update", "delete", "drop", "alter", "create", "attach", "copy"]
    if any(k in s for k in bad):
        return False
    if TABLE_NAME.lower() not in s:
        return False
    return True


def generate_sql(question: str) -> str:
    prompt = SQL_PROMPT.format(schema=FIN_SCHEMA, q=question, table=TABLE_NAME)
    txt = chat_completion([{"role": "user", "content": prompt}], temperature=0.0)

    obj = _safe_json_extract(txt)
    sql = obj.get("sql", "").strip()

    if not sql:
        raise ValueError(f"Model did not return sql. Raw output: {txt[:200]}")

    if not is_safe_sql(sql):
        raise ValueError(f"Unsafe or invalid SQL generated: {sql}")

    return sql


def repair_sql(question: str, bad_sql: str, error_msg: str) -> str:
    prompt = REPAIR_PROMPT.format(
        schema=FIN_SCHEMA,
        q=question,
        bad_sql=bad_sql,
        err=error_msg,
        table=TABLE_NAME,
    )
    txt = chat_completion([{"role": "user", "content": prompt}], temperature=0.0)

    obj = _safe_json_extract(txt)
    sql = obj.get("sql", "").strip()

    if not sql:
        raise ValueError(f"Model did not return sql in repair. Raw output: {txt[:200]}")

    if not is_safe_sql(sql):
        raise ValueError(f"Unsafe SQL after repair: {sql}")

    return sql
