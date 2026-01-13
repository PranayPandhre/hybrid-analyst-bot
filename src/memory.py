import re

TICKERS = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META"]

def extract_ticker(text: str):
    for t in TICKERS:
        if re.search(rf"\b{t}\b", text.upper()):
            return t
    return None

def resolve_followup(question: str, memory: dict) -> str:
    # If user didn't mention a company/ticker, attach the last one
    if extract_ticker(question) is None and memory.get("last_ticker"):
        return f"{question} (Company ticker context: {memory['last_ticker']})"
    return question

def infer_ticker_from_name(con, name: str) -> str | None:
    q = f"""
    SELECT ticker
    FROM financial_overview
    WHERE company_name ILIKE '%{name}%'
    LIMIT 1;
    """
    try:
        df = con.execute(q).fetchdf()
        if len(df) > 0:
            return str(df.iloc[0]["ticker"])
    except Exception:
        pass
    return None

