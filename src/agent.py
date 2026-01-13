from src.router import route_query
from src.db_sql_agent import generate_sql, repair_sql
from src.db import run_sql
from src.rag import retrieve
from src.rag_answer import answer_from_docs
from src.memory import extract_ticker, resolve_followup


def _should_force_rag(question: str) -> bool:
    q = question.lower()

    qualitative_triggers = [
        "initiative", "initiatives", "strategy", "risk", "risks", "headwinds",
        "drivers", "what drove", "why", "how", "explain", "commentary",
    ]
    numeric_triggers = [
        "market cap", "revenue", "net income", "eps", "profit", "margin",
        "compare", "top", "highest", "lowest", "billions", "$",
    ]

    has_qual = any(t in q for t in qualitative_triggers)
    has_num = any(t in q for t in numeric_triggers)

    # If it's qualitative and not explicitly numeric, force RAG.
    return has_qual and not has_num


def answer(question, state, con, vectordb):
    q2 = resolve_followup(question, state)

    # update memory
    t = extract_ticker(question)
    if t:
        state["last_ticker"] = t

    route = route_query(q2)
    r = route.get("route", "RAG")
    if r not in ("SQL", "RAG"):
        r = "RAG"


    # ✅ ADD THIS BLOCK RIGHT HERE
    if r == "BOTH" and _should_force_rag(q2):
        r = "RAG"

    if r == "SQL":
        sql = generate_sql(q2)
        try:
            df = run_sql(con, sql)
            return {
                "final": df.to_markdown(index=False),
                "trace": {"source": "db", "sql": sql, "route_reason": route.get("reason")}
            }
        except Exception as e:
            sql2 = repair_sql(q2, sql, str(e))
            df = run_sql(con, sql2)
            return {
                "final": df.to_markdown(index=False),
                "trace": {"source": "db", "sql": sql2, "route_reason": route.get("reason"), "repaired_from": sql}
            }

    if r == "RAG":
        docs = retrieve(vectordb, q2, k=4)
        ans, cites = answer_from_docs(q2, docs)
        return {
            "final": ans,
            "trace": {"source": "pdf", "citations": cites, "route_reason": route.get("reason")}
        }

    # BOTH
    sql = generate_sql(q2)
    df = run_sql(con, sql)
    docs = retrieve(vectordb, f"{q2}\nStructured result:\n{df.to_string(index=False)}", k=4)
    ans, cites = answer_from_docs(q2, docs)
    return {
        "final": f"**Database result:**\n{df.to_markdown(index=False)}\n\n**Document insight:**\n{ans}",
        "trace": {"source": "both", "sql": sql, "citations": cites, "route_reason": route.get("reason")}
    }
