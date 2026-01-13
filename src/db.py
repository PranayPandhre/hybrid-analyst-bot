import duckdb
import pandas as pd

TABLE_NAME = "financial_overview"

def init_duckdb(csv_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    df = pd.read_csv(csv_path)
    con.register("tmp_df", df)
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM tmp_df;")
    return con

def run_sql(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()
