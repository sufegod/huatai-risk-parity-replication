"""
对比回测：中证1000 vs 中证500（向量化简化版）
"""
import os, sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize
import pyodbc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "数据" / "日度收益数据更新"
OUTPUT_DIR = PROJECT_ROOT / "策略复现与回测" / "每日更新策略" / "输出" / "对比分析"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MPLCONFIG_DIR = Path(__file__).parent / ".matplotlib"
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =========== 连接数据库，拉取中证500数据 ===========
print("正在从 JYDB 拉取中证500(IC)主连前复权数据...")
conn_str = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=192.168.10.48,1433;"
    "DATABASE=JYDB;"
    "UID=tsreadonly;"
    "PWD=tstonero26*;"
    "Encrypt=no;TrustServerCertificate=yes"
)

def fetch_ic_quotes(conn, start_date="2013-01-01", end_date="2026-06-05"):
    cursor = conn.cursor()
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
  AND TradingDay BETWEEN ? AND ?
  AND ClosePrice IS NOT NULL
ORDER BY TradingDay, ContractInnerCode
""", start_date, end_date)
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    df = pd.DataFrame([dict(zip(cols, r)) for r in rows])
    df["日期"] = pd.to_datetime(df["日期"])
    df["收盘价"] = pd.to_numeric(df["收盘价"], errors="coerce")
    df["主力标志"] = pd.to_numeric(df["主力标志"], errors="coerce")
    return df.dropna(subset=["日期", "合约内部编码", "收盘价"]).sort_values(["日期", "合约内部编码"])

def build_adjusted_main_price(quotes: pd.DataFrame, asset_name: str) -> pd.Series:
    main = quotes[quotes["主力标志"] == 1].copy()
    main = main.sort_values("日期").drop_duplicates("日期", keep="last")
    main = main.set_index("日期")
    contracts_by_date = main["合约内部编码"]
    price_by_date = main["收盘价"]
    all_contracts = quotes.pivot(index="日期", columns="合约内部编码", values="收盘价")

    adj_prices = []
    cum_adj = 1.0
    dates = main.index.sort_values()

    for i, date in enumerate(dates):
        adj_prices.append((date, price_by_date[date] * cum_adj))
        if i + 1 < len(dates):
            next_date = dates[i + 1]
            cur_contract = contracts_by_date[date]
            next_contract = contracts_by_date[next_date]
            if cur_contract != next_contract:
                try:
                    old_p = all_contracts.loc[next_date, cur_contract]
                    new_p = all_contracts.loc[next_date, next_contract]
                    if pd.notna(old_p) and pd.notna(new_p) and new_p != 0:
                        cum_adj *= old_p / new_p
                except KeyError:
                    pass

    result = pd.Series(dict(adj_prices), name=asset_name)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()

# =========== 加载数据 ===========
print("加载现有日涨跌幅数据...")
df_filled = pd.read_csv(DATA_DIR / "日涨跌幅_填充.csv", index_col="日期", parse_dates=True)
df_unfilled = pd.read_csv(DATA_DIR / "日涨跌幅_未填充.csv", index_col="日期", parse_dates=True)

conn = pyodbc.connect(conn_str, timeout=30)
ic_quotes = fetch_ic_quotes(conn, start_date="2013-01-01", end_date="2026-06-05")
conn.close()
print(f"IC行情拉取成功: {len(ic_quotes)} 行, {ic_quotes['日期'].min().date()} ~ {ic_quotes['日期'].max().date()}")

ic_price = build_adjusted_main_price(ic_quotes, "中证500主连")
ic_returns = ic_price.pct_change() * 100.0

# 构建中证500版数据集
df_filled_500 = df_filled.copy()
df_filled_500["中证500主连"] = ic_returns.reindex(df_filled_500.index).fillna(0)
df_filled_500 = df_filled_500.drop(columns=["中证1000主连"])

df_unfilled_500 = df_unfilled.copy()
df_unfilled_500["中证500主连"] = ic_returns.reindex(df_unfilled_500.index)
df_unfilled_500 = df_unfilled_500.drop(columns=["中证1000主连"])

# =========== 策略参数 ===========
EWMA_DECAY = 0.97
INDEX_BASE_WEIGHT = 0.30
FEE_RATE = 0.0005
WINDOW = 252

MARGIN = {
    '沪深300主连': 0.14, '中证1000主连': 0.14, '中证500主连': 0.14,
    '红利低波ETF': 1.00, '10年国债主连': 0.025, '30年国债主连': 0.05,
    '沪铜主连': 0.16, '沪铝主连': 0.16, 'PTA主连': 0.17,
    '原油主连': 0.32, '豆粕主连': 0.13, '沪金主连': 0.28,
}

FILE_INDEX_SIGNAL = PROJECT_ROOT / "数据" / "原始数据" / "股指期货信号.xlsx"

def load_signal():
    df = pd.read_excel(FILE_INDEX_SIGNAL, header=0, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors='coerce')
    df = df[df.index.notna()]
    return df.iloc[:, 0].sort_index()

def ewma_semi_cov(df, decay=0.97):
    d = np.minimum(df.values, 0.0)
    T, N = d.shape
    w = decay ** np.arange(T - 1, -1, -1)
    w /= w.sum()
    wd = d * np.sqrt(w[:, None])
    return wd.T @ wd * 252 + np.eye(N) * 1e-8

def risk_parity_weights(cov):
    n = cov.shape[0]
    def obj(x): return 0.5 * x @ cov @ x - np.sum(np.log(x)) / n
    def jac(x): return cov @ x - 1.0 / (n * x)
    res = minimize(obj, np.ones(n), method='L-BFGS-B', jac=jac,
                   bounds=[(1e-8, None)] * n, options={'ftol': 1e-12})
    w = res.x / res.x.sum()
    return w

def run_backtest(df_w, df_t, idx_assets, rp_assets, label):
    print(f"\n{'='*50}\n回测: {label}")
    signal_series = load_signal()
    all_assets = rp_assets + idx_assets
    dates = df_w.index
    nav_vals = np.ones(len(dates))

    for i in range(WINDOW, len(dates)):
        date = dates[i]
        # 信号
        past = signal_series[signal_series.index <= date]
        sig = float(past.iloc[-1]) if len(past) > 0 else 0.3
        sig = max(0.0, min(1.0, sig))

        # 风险平价权重
        window_data = df_w.iloc[i - WINDOW: i][rp_assets].fillna(0)
        cov = ewma_semi_cov(window_data)
        rp_w = risk_parity_weights(cov)

        # 组合权重
        total_idx = INDEX_BASE_WEIGHT * sig
        w_rp = rp_w * (1 - total_idx)
        w_idx = np.full(len(idx_assets), total_idx / len(idx_assets))
        weights = np.concatenate([w_rp, w_idx])
        w_dict = dict(zip(all_assets, weights))

        # 前一期权重（i=WINDOW 时从0开始）
        if i == WINDOW:
            prev_w = {a: 0.0 for a in all_assets}
        else:
            prev_w = prev_w_dict  # noqa

        # 换仓成本
        turnover = sum(abs(w_dict.get(a, 0) - prev_w.get(a, 0)) for a in all_assets)
        fee = turnover * FEE_RATE

        # 组合收益
        today_ret = df_t.iloc[i]
        port_ret = 0.0
        for a, w in w_dict.items():
            r = today_ret.get(a, 0.0)
            if pd.isna(r):
                r = 0.0
            port_ret += w * r / 100.0

        nav_vals[i] = nav_vals[i - 1] * (1 + port_ret - fee)
        prev_w_dict = w_dict.copy()

    nav_s = pd.Series(nav_vals, index=dates)[WINDOW:]
    ret_s = nav_s.pct_change().fillna(0)

    ann_ret = (nav_s.iloc[-1] ** (252 / len(nav_s)) - 1) * 100
    ann_vol = ret_s.std() * np.sqrt(252) * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = ((nav_s / nav_s.cummax()) - 1).min() * 100
    win_rate = (nav_s.resample('ME').last().pct_change().dropna() > 0).mean() * 100

    print(f"  期末净值: {nav_s.iloc[-1]:.4f}")
    print(f"  年化收益: {ann_ret:.2f}%")
    print(f"  年化波动: {ann_vol:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  月度胜率: {win_rate:.2f}%")

    return nav_s, {
        "版本": label,
        "期末净值": round(nav_s.iloc[-1], 4),
        "年化收益(%)": round(ann_ret, 2),
        "年化波动(%)": round(ann_vol, 2),
        "夏普比率": round(sharpe, 2),
        "最大回撤(%)": round(max_dd, 2),
        "月度胜率(%)": round(win_rate, 2),
    }

# =========== 资产定义 ===========
RP_ASSETS = ['红利低波ETF', '10年国债主连', '30年国债主连',
             '沪铜主连', '沪铝主连', 'PTA主连', '原油主连', '豆粕主连', '沪金主连']
IDX_1000 = ['沪深300主连', '中证1000主连']
IDX_500  = ['沪深300主连', '中证500主连']

nav_1000, m1000 = run_backtest(df_filled, df_unfilled, IDX_1000, RP_ASSETS, "原版（含中证1000）")
nav_500,  m500  = run_backtest(df_filled_500, df_unfilled_500, IDX_500, RP_ASSETS, "替换版（含中证500）")

# =========== 对比输出 ===========
print("\n" + "="*60)
print("📊 对比结果汇总")
print("="*60)
df_metrics = pd.DataFrame([m1000, m500]).set_index("版本")
print(df_metrics.to_string())
df_metrics.to_csv(OUTPUT_DIR / "中证1000_vs_中证500_指标对比.csv", encoding="utf-8-sig")

# =========== 对比图 ===========
common = nav_1000.index.intersection(nav_500.index)
n1 = nav_1000.reindex(common)
n5 = nav_500.reindex(common)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].plot(n1, label=f"含中证1000（净值 {m1000['期末净值']:.4f}）", lw=1.5, color='#1f77b4')
axes[0].plot(n5, label=f"含中证500（净值 {m500['期末净值']:.4f}）", lw=1.5, color='#ff7f0e', ls='--')
axes[0].set_title("净值走势对比：中证1000 vs 中证500", fontsize=14)
axes[0].legend(fontsize=11)
axes[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
axes[0].grid(True, alpha=0.3)

diff = (n5 / n1 - 1) * 100
axes[1].fill_between(diff.index, diff, 0, where=(diff >= 0), alpha=0.5, color='#d62728', label='中证500更优')
axes[1].fill_between(diff.index, diff, 0, where=(diff < 0), alpha=0.5, color='#2ca02c', label='中证1000更优')
axes[1].axhline(0, color='black', lw=0.8)
axes[1].set_title("相对净值差异（中证500版 / 中证1000版 - 1）%", fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
chart_path = OUTPUT_DIR / "中证1000_vs_中证500_净值对比.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n对比图已保存: {chart_path}")
print("✅ 对比回测完成！")
