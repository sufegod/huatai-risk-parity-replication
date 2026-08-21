"""
查询中证500(IC)期货的 ExchangeCode 和 OptionCode
用于复制 FuturesAsset 配置
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

# 查中证1000(IM)已知 ExchangeCode=20, OptionCode=39144 的上市所ExchangeCode
# 同理找中证500(IC) 的 OptionCode
# 先查有哪些 OptionCode 对应 IC 合约
cursor.execute("""
SELECT TOP 10 ExchangeCode, OptionCode, ContractCode, TradingDay, ClosePrice, MainContractMark
FROM dbo.Fut_TradingQuote
WHERE ContractCode LIKE 'IC%'
ORDER BY TradingDay DESC
""")
rows = cursor.fetchall()
cols = [d[0] for d in cursor.description]
print("IC近期行情样本:")
print("  " + "  ".join(cols))
for r in rows[:5]:
    print("  " + "  ".join(str(x) for x in r))

if rows:
    ex_code = rows[0][0]
    # 查 IC 所有不同 OptionCode
    cursor.execute("""
    SELECT DISTINCT OptionCode
    FROM dbo.Fut_TradingQuote
    WHERE ContractCode LIKE 'IC%'
    """)
    opts = [r[0] for r in cursor.fetchall()]
    print(f"\nIC OptionCode 列表: {opts}")
    print(f"IC ExchangeCode: {ex_code}")

# 确认中证500主连的 OptionCode 是多少
# 中证500 IC 上市于2015年4月，标的指数代码是000905
cursor.execute("""
SELECT TOP 5 ExchangeCode, OptionCode, ContractCode, TradingDay, MainContractMark
FROM dbo.Fut_TradingQuote
WHERE ContractCode LIKE 'IC%'
AND MainContractMark = 1
ORDER BY TradingDay DESC
""")
rows2 = cursor.fetchall()
cols2 = [d[0] for d in cursor.description]
print("\nIC主连合约（MainContractMark=1）近期:")
for r in rows2[:5]:
    print("  ", dict(zip(cols2, r)))

conn.close()
