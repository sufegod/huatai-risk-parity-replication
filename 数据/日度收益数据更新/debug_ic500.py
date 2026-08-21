"""
调试：查看 IC 行情数据结构
"""
import pyodbc
import pandas as pd

conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.10.48,1433;"
    "DATABASE=JYDB;"
    "UID=tsreadonly;"
    "PWD=tstonero26*;"
    "Encrypt=no;TrustServerCertificate=yes"
)
conn = pyodbc.connect(conn_str, timeout=15)
cursor = conn.cursor()

cursor.execute("""
SELECT TOP 5
    TradingDay AS 日期,
    ContractInnerCode AS 合约内部编码,
    ContractCode AS 合约代码,
    CAST(ClosePrice AS float) AS 收盘价,
    CAST(MainContractMark AS int) AS 主力标志
FROM dbo.Fut_TradingQuote
WHERE ExchangeCode = 20
  AND OptionCode = 4978
  AND TradingDay >= '2026-06-01'
  AND ClosePrice IS NOT NULL
ORDER BY TradingDay, ContractInnerCode
""")
rows = cursor.fetchall()
cols = [d[0] for d in cursor.description]
print("列数:", len(cols), "列名:", cols)
print("行数:", len(rows))
for r in rows:
    print(dict(zip(cols, r)))

# 全量拉取，看看 DataFrame 的 shape
cursor.execute("""
SELECT
    TradingDay AS 日期,
    ContractInnerCode AS 合约内部编码,
    ContractCode AS 合约代码,
    CAST(ClosePrice AS float) AS 收盘价,
    CAST(MainContractMark AS int) AS 主力标志
FROM dbo.Fut_TradingQuote
WHERE ExchangeCode = 20
  AND OptionCode = 4978
  AND TradingDay BETWEEN '2013-01-01' AND '2026-06-05'
  AND ClosePrice IS NOT NULL
ORDER BY TradingDay, ContractInnerCode
""")
all_rows = cursor.fetchall()
print(f"\n全量查询行数: {len(all_rows)}")
df = pd.DataFrame(all_rows, columns=cols)
print("DataFrame shape:", df.shape)
print(df.head(3))
conn.close()
