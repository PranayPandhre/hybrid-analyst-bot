from src.db import init_duckdb, run_sql, TABLE_NAME

con = init_duckdb("data/financial_data.csv")

# 1) Simple lookup
df = run_sql(con, f"""
SELECT company_name, ticker, revenue_2023_billions
FROM {TABLE_NAME}
WHERE ticker = 'MSFT';
""")
print("=== MSFT revenue ===")
print(df.to_string(index=False))

# 2) Compare Apple vs Microsoft
df2 = run_sql(con, f"""
SELECT company_name, ticker, revenue_2023_billions, net_income_2023_billions
FROM {TABLE_NAME}
WHERE ticker IN ('AAPL', 'MSFT')
ORDER BY revenue_2023_billions DESC;
""")
print("\n=== AAPL vs MSFT comparison ===")
print(df2.to_string(index=False))
