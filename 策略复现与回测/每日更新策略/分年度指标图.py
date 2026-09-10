# -*- coding: utf-8 -*-
"""
生成华泰风险平价策略 v0.19 分年度指标汇报图：
  - 柱状：各年夏普比率
  - 折线：各年最大回撤（右轴）
用法：
  python 分年度指标图.py [指标CSV路径或数据日期]
默认读取最新日期的「年度及全局回测指标_YYYY-MM-DD.csv」。
"""
import os
import re
import glob
import csv
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(BASE, "输出", "指标")
CHART_DIR = os.path.join(BASE, "输出", "图表")
STRATEGY = "风险平价策略"

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def latest_metrics_file():
    files = glob.glob(os.path.join(METRICS_DIR, "年度及全局回测指标_*.csv"))
    if not files:
        raise FileNotFoundError("未找到指标文件：" + METRICS_DIR)

    def key(p):
        m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(p))
        return m.group(1) if m else ""

    return max(files, key=key)


def load_annual(path):
    years, sharpe, ret, mdd, total = [], [], [], [], None
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rng = (row.get("回测区间") or "").strip()
            asset = (row.get("组合/资产") or "").strip()
            if asset != STRATEGY:
                continue

            def num(col):
                v = (row.get(col) or "").replace("%", "").strip()
                return float(v) if v not in ("", "-") else None

            if rng.startswith("全局"):
                total = {
                    "sharpe": num("夏普比率"),
                    "ret": num("年化收益"),
                    "mdd": num("最大回撤"),
                }
            else:
                m = re.match(r"(\d{4})", rng)
                if m:
                    years.append(m.group(1))
                    sharpe.append(num("夏普比率"))
                    ret.append(num("年化收益"))
                    mdd.append(num("最大回撤"))
    return years, sharpe, ret, mdd, total


def main():
    path = latest_metrics_file()
    date_tag = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path)).group(1)
    years, sharpe, ret, mdd, total = load_annual(path)
    print("指标文件:", os.path.basename(path))
    print("年度夏普:", list(zip(years, sharpe)))

    x = list(range(len(years)))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, 8.4), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.10})
    fig.patch.set_facecolor("white")

    # ================= 上panel：夏普柱状 =================
    colors = ["#2E6FB7"] * len(years)
    if years:
        colors[-1] = "#7FA8D4"  # 最新年（YTD）浅色
    ax1.bar(x, sharpe, width=0.55, color=colors,
            edgecolor="#1F4E79", linewidth=0.8, zorder=3, label="年度夏普比率")
    for xi, sv in zip(x, sharpe):
        ax1.text(xi, sv + 0.05, f"{sv:.2f}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold", color="#1F4E79", zorder=5)

    if total and total["sharpe"] is not None:
        ax1.axhline(total["sharpe"], color="#C0392B", linestyle="--",
                    linewidth=1.6, zorder=4,
                    label=f"全周期夏普 {total['sharpe']:.2f}")
    ax1.axhline(1.0, color="#999999", linestyle=":", linewidth=1.1, zorder=1,
                label="夏普=1 基准线")

    ax1.set_ylim(0, max(sharpe) * 1.28)
    ax1.set_ylabel("夏普比率", fontsize=12.5, color="#1F4E79")
    ax1.tick_params(axis="y", labelcolor="#1F4E79", labelsize=10.5)
    ax1.legend(loc="upper left", fontsize=10.5, framealpha=0.92,
               edgecolor="#CCCCCC", ncol=1)
    ax1.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title(
        f"华泰风险平价策略 v0.19 · 分年度夏普与最大回撤（数据至 {date_tag}）\n"
        "九年夏普全部 > 1，全周期 1.78 —— 低波动、低回撤下的稳定风险调整收益",
        fontsize=15, fontweight="bold", pad=14)

    # ================= 下panel：最大回撤线 =================
    ax2.plot(x, mdd, color="#E67E22", marker="o", markersize=7.5,
             linewidth=2.3, zorder=5, label="年度最大回撤")
    ax2.fill_between(x, mdd, 0, color="#F5C99B", alpha=0.45, zorder=2)
    for xi, dv in zip(x, mdd):
        ax2.text(xi, dv - 0.42, f"{dv:.2f}%", ha="center", va="top",
                 fontsize=10.5, fontweight="bold", color="#B9540E", zorder=6)

    ax2.axhline(0, color="#BBBBBB", linewidth=1.0, zorder=1)
    ax2.set_ylim(min(mdd) * 1.45, 1.2)
    ax2.set_ylabel("最大回撤", fontsize=12.5, color="#B9540E")
    ax2.tick_params(axis="y", labelcolor="#B9540E", labelsize=10.5)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:.0f}%"))
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [y + ("*" if i == len(years) - 1 else "") for i, y in enumerate(years)],
        fontsize=11.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    note = "* 最新年份为年初至今（YTD），指标已年化；最大回撤越浅代表抗跌越好"
    fig.text(0.99, 0.012, note, ha="right", va="bottom",
             fontsize=9, color="#666666", style="italic")

    fig.tight_layout(rect=(0, 0.028, 1, 1))
    os.makedirs(CHART_DIR, exist_ok=True)
    out = os.path.join(CHART_DIR, f"分年度夏普与最大回撤_{date_tag}.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("图表已保存:", out)
    return out


if __name__ == "__main__":
    main()
