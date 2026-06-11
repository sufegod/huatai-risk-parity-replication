from __future__ import annotations

import html
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CYCLE_PATH = SCRIPT_DIR / "国信证券-货币信用划分.xlsx"
RETURN_PATH = PROJECT_ROOT / "数据" / "日度收益数据更新" / "日涨跌幅_填充.csv"
OUTPUT_DIR = SCRIPT_DIR / "分析结果"

REPO_COLUMN = "一天期国债逆回购"
DATE_COLUMN = "日期"
CYCLE_COLUMN = "周期划分"
MONTH_COLUMN = "月份"
RETURN_COLUMN = "月收益(%)"

ALL_CYCLES = ["宽货币宽信用", "宽货币紧信用", "紧货币宽信用", "紧货币紧信用"]
MAIN_TEST_CYCLES = ["宽货币紧信用", "宽货币宽信用", "紧货币紧信用"]
EXPECTED_CYCLE_COUNTS = {
    "宽货币紧信用": 69,
    "宽货币宽信用": 48,
    "紧货币紧信用": 14,
    "紧货币宽信用": 1,
}

FIG_BOX = "三类有效象限资产月收益箱线图.png"
FIG_HEATMAP = "四象限资产年化收益热力图.png"
FIG_SIG = "显著性检验摘要图.png"
FIG_POOL = "三周期优选资产池.png"
UNDERPERFORMANCE_THRESHOLD_PP = 10.0


def configure_plot_style() -> None:
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#FCFCFD"
    plt.rcParams["axes.facecolor"] = "#FFFFFF"
    plt.rcParams["axes.edgecolor"] = "#D7DBE7"
    plt.rcParams["axes.labelcolor"] = "#1F2430"
    plt.rcParams["xtick.color"] = "#464C55"
    plt.rcParams["ytick.color"] = "#464C55"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_cycle_data() -> pd.DataFrame:
    cycle = pd.read_excel(CYCLE_PATH)
    cycle = cycle.rename(columns={cycle.columns[0]: DATE_COLUMN})
    required = {DATE_COLUMN, CYCLE_COLUMN}
    missing = required - set(cycle.columns)
    if missing:
        raise ValueError(f"货币信用划分表缺少必要列: {sorted(missing)}")

    cycle[DATE_COLUMN] = pd.to_datetime(cycle[DATE_COLUMN])
    cycle[MONTH_COLUMN] = cycle[DATE_COLUMN].dt.to_period("M").astype(str)
    cycle[CYCLE_COLUMN] = cycle[CYCLE_COLUMN].astype(str).str.strip()
    cycle = cycle.dropna(subset=[DATE_COLUMN, CYCLE_COLUMN])
    duplicated = cycle[cycle[MONTH_COLUMN].duplicated()][MONTH_COLUMN].tolist()
    if duplicated:
        raise ValueError(f"货币信用划分表存在重复月份: {duplicated[:10]}")
    return cycle


def read_daily_returns() -> tuple[pd.DataFrame, list[str]]:
    daily = pd.read_csv(RETURN_PATH, encoding="utf-8-sig")
    if DATE_COLUMN not in daily.columns:
        raise ValueError(f"日收益文件缺少 `{DATE_COLUMN}` 列")

    daily[DATE_COLUMN] = pd.to_datetime(daily[DATE_COLUMN])
    daily[MONTH_COLUMN] = daily[DATE_COLUMN].dt.to_period("M").astype(str)
    assets = [c for c in daily.columns if c not in {DATE_COLUMN, MONTH_COLUMN, REPO_COLUMN}]
    if not assets:
        raise ValueError("日收益文件没有可分析的资产列")

    daily[assets] = daily[assets].apply(pd.to_numeric, errors="coerce")
    return daily, assets


def compound_monthly_returns(daily: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    factors = daily[[MONTH_COLUMN, *assets]].copy()
    factors[assets] = 1.0 + factors[assets] / 100.0
    monthly = (factors.groupby(MONTH_COLUMN, sort=True)[assets].prod(min_count=1) - 1.0) * 100.0
    return monthly.reset_index()


def build_monthly_panel(cycle: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    panel = monthly.merge(cycle[[MONTH_COLUMN, CYCLE_COLUMN]], on=MONTH_COLUMN, how="inner")
    panel[CYCLE_COLUMN] = pd.Categorical(panel[CYCLE_COLUMN], categories=ALL_CYCLES, ordered=True)
    panel = panel.sort_values(MONTH_COLUMN).reset_index(drop=True)
    return panel


def to_long_panel(panel: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    long_panel = panel.melt(
        id_vars=[MONTH_COLUMN, CYCLE_COLUMN],
        value_vars=assets,
        var_name="资产",
        value_name=RETURN_COLUMN,
    )
    long_panel[CYCLE_COLUMN] = long_panel[CYCLE_COLUMN].astype(str)
    return long_panel.sort_values([MONTH_COLUMN, "资产"]).reset_index(drop=True)


def annualized_return(monthly_returns: pd.Series) -> float:
    values = monthly_returns.dropna().to_numpy(dtype=float) / 100.0
    if len(values) == 0:
        return np.nan
    total = float(np.prod(1.0 + values))
    if total <= 0:
        return np.nan
    return (total ** (12.0 / len(values)) - 1.0) * 100.0


def describe_by_cycle(long_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = long_panel.groupby(["资产", CYCLE_COLUMN], observed=False)
    for (asset, cycle), group in grouped:
        series = group[RETURN_COLUMN].dropna().astype(float)
        n = int(series.shape[0])
        rows.append(
            {
                "资产": asset,
                "周期划分": cycle,
                "样本月数": n,
                "样本用途": "主检验" if cycle in MAIN_TEST_CYCLES and n >= 2 else "仅描述",
                "月均收益(%)": series.mean() if n else np.nan,
                "月收益中位数(%)": series.median() if n else np.nan,
                "月收益标准差(%)": series.std(ddof=1) if n > 1 else np.nan,
                "年化收益(%)": annualized_return(series),
                "年化波动率(%)": series.std(ddof=1) * math.sqrt(12.0) if n > 1 else np.nan,
                "胜率(%)": (series.gt(0).mean() * 100.0) if n else np.nan,
                "25%分位数(%)": series.quantile(0.25) if n else np.nan,
                "75%分位数(%)": series.quantile(0.75) if n else np.nan,
                "最小月收益(%)": series.min() if n else np.nan,
                "最大月收益(%)": series.max() if n else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result[CYCLE_COLUMN] = pd.Categorical(result[CYCLE_COLUMN], categories=ALL_CYCLES, ordered=True)
    return result.sort_values(["资产", CYCLE_COLUMN]).reset_index(drop=True)


def classify_asset_cycle_pools(
    description: pd.DataFrame,
    threshold_pp: float = UNDERPERFORMANCE_THRESHOLD_PP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    main = description[description[CYCLE_COLUMN].isin(MAIN_TEST_CYCLES)].copy()
    for asset, group in main.groupby("资产", sort=False):
        group = group.set_index(CYCLE_COLUMN).reindex(MAIN_TEST_CYCLES)
        annual_returns = group["年化收益(%)"].astype(float)
        best_cycle = annual_returns.idxmax()
        best_return = float(annual_returns.loc[best_cycle])
        median_return = float(annual_returns.median())

        for cycle in MAIN_TEST_CYCLES:
            annual_return = float(annual_returns.loc[cycle])
            gap = best_return - annual_return
            remove = bool(cycle != best_cycle and gap >= threshold_pp and annual_return <= median_return)
            rows.append(
                {
                    "资产": asset,
                    "周期划分": cycle,
                    "年化收益(%)": annual_return,
                    "月均收益(%)": float(group.loc[cycle, "月均收益(%)"]),
                    "胜率(%)": float(group.loc[cycle, "胜率(%)"]),
                    "该资产最佳周期": best_cycle,
                    "该资产最佳周期年化收益(%)": best_return,
                    "与最佳周期差距(百分点)": gap,
                    "三周期年化收益中位数(%)": median_return,
                    "明显较差判定阈值(百分点)": threshold_pp,
                    "是否剔除": remove,
                    "剔除原因": (
                        f"较最佳周期低 {gap:.1f} 个百分点，且不高于三周期中位数"
                        if remove
                        else "保留"
                    ),
                }
            )

    detail = pd.DataFrame(rows)
    detail[CYCLE_COLUMN] = pd.Categorical(detail[CYCLE_COLUMN], categories=MAIN_TEST_CYCLES, ordered=True)
    detail = detail.sort_values(["资产", CYCLE_COLUMN]).reset_index(drop=True)

    pool_rows = []
    for cycle in MAIN_TEST_CYCLES:
        kept = detail[(detail[CYCLE_COLUMN] == cycle) & (~detail["是否剔除"])].copy()
        removed = detail[(detail[CYCLE_COLUMN] == cycle) & (detail["是否剔除"])].copy()
        removed = removed.sort_values(["与最佳周期差距(百分点)", "年化收益(%)"], ascending=[False, True])
        pool_rows.append(
            {
                "周期划分": cycle,
                "入选资产数": int(len(kept)),
                "剔除资产数": int(len(removed)),
                "资产池": "、".join(kept["资产"].tolist()),
                "剔除资产": "、".join(removed["资产"].tolist()),
                "平均年化收益(%)": float(kept["年化收益(%)"].mean()) if len(kept) else np.nan,
                "平均月均收益(%)": float(kept["月均收益(%)"].mean()) if len(kept) else np.nan,
                "平均胜率(%)": float(kept["胜率(%)"].mean()) if len(kept) else np.nan,
            }
        )

    pools = pd.DataFrame(pool_rows)
    pools[CYCLE_COLUMN] = pd.Categorical(pools[CYCLE_COLUMN], categories=MAIN_TEST_CYCLES, ordered=True)
    return detail, pools.sort_values(CYCLE_COLUMN).reset_index(drop=True)


def bh_fdr(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    finite_p = p[finite_mask]
    m = len(finite_p)
    if m == 0:
        return adjusted.tolist()

    order = np.argsort(finite_p)
    ranked = finite_p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)
    restored = np.empty_like(ranked)
    restored[order] = ranked
    adjusted[finite_mask] = restored
    return adjusted.tolist()


def welch_anova(groups: list[np.ndarray]) -> tuple[float, float]:
    clean_groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(clean_groups) < 2:
        return np.nan, np.nan

    try:
        result = stats.f_oneway(*clean_groups, equal_var=False)
        return float(result.statistic), float(result.pvalue)
    except TypeError:
        pass

    sizes = np.asarray([len(g) for g in clean_groups], dtype=float)
    means = np.asarray([np.mean(g) for g in clean_groups], dtype=float)
    variances = np.asarray([np.var(g, ddof=1) for g in clean_groups], dtype=float)
    if np.any(variances <= 0) or np.any(~np.isfinite(variances)):
        return np.nan, np.nan

    weights = sizes / variances
    weight_sum = weights.sum()
    weighted_mean = np.sum(weights * means) / weight_sum
    k = len(clean_groups)
    numerator = np.sum(weights * (means - weighted_mean) ** 2) / (k - 1)
    correction_sum = np.sum((1.0 / (sizes - 1.0)) * (1.0 - weights / weight_sum) ** 2)
    denominator = 1.0 + (2.0 * (k - 2.0) / (k**2 - 1.0)) * correction_sum
    f_stat = numerator / denominator
    df1 = k - 1
    df2 = (k**2 - 1.0) / (3.0 * correction_sum)
    p_value = float(stats.f.sf(f_stat, df1, df2))
    return float(f_stat), p_value


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) == 0 or len(right) == 0:
        return np.nan
    diff = left[:, None] - right[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def effect_label(delta: float) -> str:
    if not np.isfinite(delta):
        return "NA"
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "极小"
    if abs_delta < 0.33:
        return "小"
    if abs_delta < 0.474:
        return "中"
    return "大"


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "不可检验"
    if p_value <= 0.05:
        return "显著"
    if p_value <= 0.10:
        return "边际显著"
    return "不显著"


def run_significance_tests(long_panel: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    overall_rows = []
    pairwise_candidates: dict[str, bool] = {}

    for asset in assets:
        asset_data = long_panel[long_panel["资产"] == asset]
        groups = [
            asset_data.loc[asset_data[CYCLE_COLUMN] == cycle, RETURN_COLUMN].dropna().to_numpy(dtype=float)
            for cycle in MAIN_TEST_CYCLES
        ]
        group_sizes = {f"{cycle}样本月数": len(group) for cycle, group in zip(MAIN_TEST_CYCLES, groups)}

        if all(len(group) >= 2 for group in groups):
            kruskal = stats.kruskal(*groups)
            welch_f, welch_p = welch_anova(groups)
            means = [float(np.mean(group)) for group in groups]
            best_idx = int(np.argmax(means))
            worst_idx = int(np.argmin(means))
            range_mean = means[best_idx] - means[worst_idx]
            overall_rows.append(
                {
                    "结果类型": "overall",
                    "资产": asset,
                    "比较": "三类有效象限整体差异",
                    **group_sizes,
                    "Kruskal统计量": float(kruskal.statistic),
                    "Kruskal_p值": float(kruskal.pvalue),
                    "Kruskal_FDR": np.nan,
                    "Welch_F统计量": welch_f,
                    "Welch_p值": welch_p,
                    "Welch_FDR": np.nan,
                    "组A": "",
                    "组B": "",
                    "组A均值(%)": np.nan,
                    "组B均值(%)": np.nan,
                    "均值差_组A减组B(百分点)": np.nan,
                    "Cliffs_delta_组A相对组B": np.nan,
                    "效应量等级": "",
                    "表现最好象限_按月均收益": MAIN_TEST_CYCLES[best_idx],
                    "表现最弱象限_按月均收益": MAIN_TEST_CYCLES[worst_idx],
                    "最大月均收益差(百分点)": range_mean,
                    "结论": "",
                }
            )
        else:
            overall_rows.append(
                {
                    "结果类型": "overall",
                    "资产": asset,
                    "比较": "三类有效象限整体差异",
                    **group_sizes,
                    "Kruskal统计量": np.nan,
                    "Kruskal_p值": np.nan,
                    "Kruskal_FDR": np.nan,
                    "Welch_F统计量": np.nan,
                    "Welch_p值": np.nan,
                    "Welch_FDR": np.nan,
                    "组A": "",
                    "组B": "",
                    "组A均值(%)": np.nan,
                    "组B均值(%)": np.nan,
                    "均值差_组A减组B(百分点)": np.nan,
                    "Cliffs_delta_组A相对组B": np.nan,
                    "效应量等级": "",
                    "表现最好象限_按月均收益": "",
                    "表现最弱象限_按月均收益": "",
                    "最大月均收益差(百分点)": np.nan,
                    "结论": "不可检验",
                }
            )

    overall = pd.DataFrame(overall_rows)
    overall["Kruskal_FDR"] = bh_fdr(overall["Kruskal_p值"])
    overall["Welch_FDR"] = bh_fdr(overall["Welch_p值"])
    overall["结论"] = overall["Kruskal_FDR"].map(significance_label)
    for _, row in overall.iterrows():
        pairwise_candidates[row["资产"]] = (
            np.isfinite(row["Kruskal_p值"])
            and (row["Kruskal_p值"] <= 0.10 or row["Kruskal_FDR"] <= 0.10)
        )

    pairwise_rows = []
    for asset in assets:
        if not pairwise_candidates.get(asset, False):
            continue
        asset_data = long_panel[long_panel["资产"] == asset]
        for cycle_a, cycle_b in combinations(MAIN_TEST_CYCLES, 2):
            a = asset_data.loc[asset_data[CYCLE_COLUMN] == cycle_a, RETURN_COLUMN].dropna().to_numpy(dtype=float)
            b = asset_data.loc[asset_data[CYCLE_COLUMN] == cycle_b, RETURN_COLUMN].dropna().to_numpy(dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue
            u_result = stats.mannwhitneyu(a, b, alternative="two-sided")
            delta = cliffs_delta(a, b)
            pairwise_rows.append(
                {
                    "结果类型": "pairwise",
                    "资产": asset,
                    "比较": f"{cycle_a} vs {cycle_b}",
                    **{f"{cycle}样本月数": np.nan for cycle in MAIN_TEST_CYCLES},
                    "Kruskal统计量": np.nan,
                    "Kruskal_p值": float(u_result.pvalue),
                    "Kruskal_FDR": np.nan,
                    "Welch_F统计量": np.nan,
                    "Welch_p值": np.nan,
                    "Welch_FDR": np.nan,
                    "组A": cycle_a,
                    "组B": cycle_b,
                    "组A均值(%)": float(np.mean(a)),
                    "组B均值(%)": float(np.mean(b)),
                    "均值差_组A减组B(百分点)": float(np.mean(a) - np.mean(b)),
                    "Cliffs_delta_组A相对组B": delta,
                    "效应量等级": effect_label(delta),
                    "表现最好象限_按月均收益": cycle_a if np.mean(a) >= np.mean(b) else cycle_b,
                    "表现最弱象限_按月均收益": cycle_b if np.mean(a) >= np.mean(b) else cycle_a,
                    "最大月均收益差(百分点)": abs(float(np.mean(a) - np.mean(b))),
                    "结论": "",
                }
            )

    pairwise = pd.DataFrame(pairwise_rows)
    if not pairwise.empty:
        pairwise["Kruskal_FDR"] = bh_fdr(pairwise["Kruskal_p值"])
        pairwise["结论"] = pairwise["Kruskal_FDR"].map(significance_label)

    return pd.concat([overall, pairwise], ignore_index=True)


def save_csvs(
    long_panel: pd.DataFrame,
    description: pd.DataFrame,
    tests: pd.DataFrame,
    asset_pool_detail: pd.DataFrame,
    asset_pools: pd.DataFrame,
) -> None:
    long_panel.to_csv(OUTPUT_DIR / "象限月度收益明细.csv", index=False, encoding="utf-8-sig")
    description.to_csv(OUTPUT_DIR / "象限描述统计.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUTPUT_DIR / "显著性检验结果.csv", index=False, encoding="utf-8-sig")
    asset_pool_detail.to_csv(OUTPUT_DIR / "资产周期剔除明细.csv", index=False, encoding="utf-8-sig")
    asset_pools.to_csv(OUTPUT_DIR / "三周期优选资产池.csv", index=False, encoding="utf-8-sig")


def save_boxplot(long_panel: pd.DataFrame, assets: list[str]) -> None:
    data = long_panel[long_panel[CYCLE_COLUMN].isin(MAIN_TEST_CYCLES)].copy()
    palette = {
        "宽货币紧信用": "#A3BEFA",
        "宽货币宽信用": "#FFE15B",
        "紧货币紧信用": "#F0986E",
    }
    fig, axes = plt.subplots(4, 3, figsize=(16, 18), sharey=False)
    axes = axes.ravel()
    for ax, asset in zip(axes, assets):
        series_by_cycle = [
            data.loc[(data["资产"] == asset) & (data[CYCLE_COLUMN] == cycle), RETURN_COLUMN].astype(float).values
            for cycle in MAIN_TEST_CYCLES
        ]
        box = ax.boxplot(
            series_by_cycle,
            tick_labels=MAIN_TEST_CYCLES,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#1F2430", "linewidth": 1.4},
            whiskerprops={"color": "#7A828F"},
            capprops={"color": "#7A828F"},
            boxprops={"edgecolor": "#464C55"},
        )
        for patch, cycle in zip(box["boxes"], MAIN_TEST_CYCLES):
            patch.set_facecolor(palette[cycle])
            patch.set_alpha(0.72)
        ax.axhline(0, color="#464C55", linewidth=0.8, linestyle="--", alpha=0.65)
        ax.grid(axis="y", color="#E6E8F0", linestyle=":", linewidth=0.8)
        ax.set_title(asset, fontsize=12, color="#1F2430")
        ax.tick_params(axis="x", labelrotation=18)
        ax.set_ylabel("月收益(%)")
    for ax in axes[len(assets) :]:
        ax.axis("off")
    fig.suptitle("三类有效货币信用象限下的资产月收益分布", fontsize=18, color="#1F2430", y=0.995)
    fig.text(
        0.01,
        0.01,
        "注：紧货币宽信用只有 1 个月，未纳入主检验箱线图。",
        fontsize=10,
        color="#6F768A",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(OUTPUT_DIR / FIG_BOX, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(description: pd.DataFrame, assets: list[str]) -> None:
    matrix = (
        description.pivot(index="资产", columns=CYCLE_COLUMN, values="年化收益(%)")
        .reindex(index=assets, columns=ALL_CYCLES)
        .astype(float)
    )
    fig, ax = plt.subplots(figsize=(12, 9))
    vmax = np.nanmax(np.abs(matrix.to_numpy()))
    vmax = max(vmax, 1.0)
    image = ax.imshow(matrix.to_numpy(), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(ALL_CYCLES)), labels=ALL_CYCLES)
    ax.set_yticks(np.arange(len(assets)), labels=assets)
    ax.tick_params(axis="x", labelrotation=20)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            label = "NA" if pd.isna(value) else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color="#1F2430")
    ax.set_title("四象限下各资产年化收益", fontsize=16, color="#1F2430", pad=16)
    ax.set_xlabel("周期划分")
    ax.set_ylabel("资产")
    cbar = fig.colorbar(image, ax=ax, shrink=0.84)
    cbar.set_label("年化收益(%)")
    fig.text(
        0.01,
        0.01,
        "注：紧货币宽信用仅 1 个月，年化收益只作描述参考。",
        fontsize=10,
        color="#6F768A",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUTPUT_DIR / FIG_HEATMAP, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_significance_chart(tests: pd.DataFrame) -> None:
    overall = tests[tests["结果类型"] == "overall"].copy()
    overall["plot_p"] = overall["Kruskal_FDR"].clip(lower=1e-12)
    overall["-log10(FDR)"] = -np.log10(overall["plot_p"])
    overall = overall.sort_values("-log10(FDR)", ascending=True)
    colors = np.where(
        overall["Kruskal_FDR"] <= 0.05,
        "#F0986E",
        np.where(overall["Kruskal_FDR"] <= 0.10, "#FFE15B", "#A3BEFA"),
    )
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(overall["资产"], overall["-log10(FDR)"], color=colors, edgecolor="#464C55", linewidth=0.6)
    ax.axvline(-math.log10(0.05), color="#1F2430", linewidth=1.0, linestyle="--", label="FDR=0.05")
    ax.axvline(-math.log10(0.10), color="#6F768A", linewidth=1.0, linestyle=":", label="FDR=0.10")
    for y, (_, row) in enumerate(overall.iterrows()):
        fdr = row["Kruskal_FDR"]
        label = "NA" if pd.isna(fdr) else f"{fdr:.3f}"
        ax.text(row["-log10(FDR)"] + 0.02, y, label, va="center", fontsize=9, color="#464C55")
    ax.set_title("三类有效象限整体差异检验摘要", fontsize=16, color="#1F2430", pad=14)
    ax.set_xlabel("-log10(Kruskal-Wallis FDR)")
    ax.set_ylabel("资产")
    ax.grid(axis="x", color="#E6E8F0", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / FIG_SIG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def wrap_assets(text: str, per_line: int = 4) -> str:
    assets = [item for item in str(text).split("、") if item]
    lines = ["、".join(assets[i : i + per_line]) for i in range(0, len(assets), per_line)]
    return "\n".join(lines) if lines else "无"


def save_asset_pool_chart(asset_pools: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), sharey=False)
    colors = ["#A3BEFA", "#FFE15B", "#F0986E"]
    for ax, (_, row), color in zip(axes, asset_pools.iterrows(), colors):
        ax.set_facecolor("#FFFFFF")
        ax.text(
            0.5,
            0.88,
            str(row[CYCLE_COLUMN]),
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold",
            color="#1F2430",
        )
        ax.text(
            0.5,
            0.73,
            f"入选 {int(row['入选资产数'])} 个 | 剔除 {int(row['剔除资产数'])} 个",
            ha="center",
            va="center",
            fontsize=12,
            color="#464C55",
        )
        ax.text(
            0.08,
            0.55,
            "资产池",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#1F2430",
        )
        ax.text(
            0.08,
            0.39,
            wrap_assets(row["资产池"]),
            ha="left",
            va="center",
            fontsize=12,
            color="#1F2430",
            linespacing=1.65,
        )
        ax.text(
            0.08,
            0.16,
            f"剔除：{wrap_assets(row['剔除资产'], per_line=3)}",
            ha="left",
            va="center",
            fontsize=10.5,
            color="#6F768A",
            linespacing=1.45,
        )
        for spine in ax.spines.values():
            spine.set_color("#D7DBE7")
        ax.axhspan(0.80, 0.98, color=color, alpha=0.35)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle("三类有效周期下剔除明显弱势周期后的资产池", fontsize=18, color="#1F2430", y=1.02)
    fig.text(
        0.01,
        0.01,
        f"剔除规则：某资产在该周期年化收益较其三周期最佳周期低至少 {UNDERPERFORMANCE_THRESHOLD_PP:.0f} 个百分点，且不高于该资产三周期中位数。",
        fontsize=10,
        color="#6F768A",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUTPUT_DIR / FIG_POOL, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_table_html(df: pd.DataFrame, max_rows: int | None = None, float_format: str = "{:.3f}") -> str:
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    return table.to_html(index=False, classes="data-table", border=0, escape=True, float_format=lambda x: float_format.format(x))


def cycle_count_sentence(counts: dict[str, int]) -> str:
    return "、".join(f"{cycle}={counts.get(cycle, 0)}" for cycle in ["宽货币紧信用", "宽货币宽信用", "紧货币紧信用", "紧货币宽信用"])


def build_report(
    panel: pd.DataFrame,
    long_panel: pd.DataFrame,
    description: pd.DataFrame,
    tests: pd.DataFrame,
    asset_pool_detail: pd.DataFrame,
    asset_pools: pd.DataFrame,
    assets: list[str],
) -> None:
    overall = tests[tests["结果类型"] == "overall"].copy()
    pairwise = tests[tests["结果类型"] == "pairwise"].copy()
    overall = overall.sort_values(["Kruskal_FDR", "Kruskal_p值"], na_position="last")
    significant = overall[overall["Kruskal_FDR"] <= 0.05]
    marginal = overall[(overall["Kruskal_FDR"] > 0.05) & (overall["Kruskal_FDR"] <= 0.10)]
    significant_pairwise = pairwise[pairwise["Kruskal_FDR"] <= 0.10].sort_values("Kruskal_FDR")

    counts = panel[CYCLE_COLUMN].astype(str).value_counts().to_dict()
    start_month = panel[MONTH_COLUMN].min()
    end_month = panel[MONTH_COLUMN].max()

    top_desc = (
        description[description[CYCLE_COLUMN].isin(MAIN_TEST_CYCLES)]
        .sort_values(["资产", "年化收益(%)"], ascending=[True, False])
        .groupby("资产", as_index=False)
        .head(1)
        [["资产", CYCLE_COLUMN, "样本月数", "年化收益(%)", "月均收益(%)", "胜率(%)"]]
        .rename(columns={CYCLE_COLUMN: "年化收益最高的有效象限"})
    )
    top_desc = top_desc.sort_values("年化收益(%)", ascending=False)

    summary_items = []
    if not significant.empty:
        names = "、".join(significant["资产"].tolist())
        summary_items.append(f"<li><strong>FDR 5% 口径下存在显著差异的资产：</strong>{html.escape(names)}。</li>")
    else:
        summary_items.append("<li><strong>FDR 5% 口径下没有资产通过整体差异检验。</strong>这意味着在多重检验校正后，四象限信号对单资产月收益的可辨识度有限。</li>")
    if not marginal.empty:
        names = "、".join(marginal["资产"].tolist())
        summary_items.append(f"<li><strong>FDR 10% 口径下边际显著的资产：</strong>{html.escape(names)}。</li>")
    summary_items.append(
        f"<li><strong>样本结构不均衡：</strong>{html.escape(cycle_count_sentence(counts))}。其中紧货币宽信用只有 1 个月，因此只展示描述统计，不参与主显著性检验。</li>"
    )
    summary_items.append(
        f"<li><strong>检验粒度：</strong>日收益先复利为月收益，再与月度象限匹配；样本期为 {html.escape(start_month)} 至 {html.escape(end_month)}，共 {len(panel)} 个月、{len(assets)} 个风险资产。</li>"
    )

    overall_cols = [
        "资产",
        "Kruskal_p值",
        "Kruskal_FDR",
        "Welch_p值",
        "Welch_FDR",
        "表现最好象限_按月均收益",
        "表现最弱象限_按月均收益",
        "最大月均收益差(百分点)",
        "结论",
    ]
    pair_cols = [
        "资产",
        "比较",
        "Kruskal_p值",
        "Kruskal_FDR",
        "均值差_组A减组B(百分点)",
        "Cliffs_delta_组A相对组B",
        "效应量等级",
        "结论",
    ]

    pair_section = (
        make_table_html(significant_pairwise[pair_cols], max_rows=20)
        if not significant_pairwise.empty
        else "<p>没有 pairwise 比较在 FDR 10% 口径下达到显著或边际显著。</p>"
    )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>货币信用四象限资产表现显著性分析</title>
  <style>
    body {{
      margin: 0;
      background: #f4f5f7;
      color: #1f2430;
      font-family: "Microsoft YaHei", "Noto Sans CJK SC", Arial, sans-serif;
      line-height: 1.62;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 28px 56px;
      background: #ffffff;
    }}
    h1 {{ font-size: 30px; margin: 0 0 8px; }}
    h2 {{ font-size: 22px; margin-top: 34px; border-bottom: 1px solid #d7dbe7; padding-bottom: 8px; }}
    h3 {{ font-size: 17px; margin-top: 24px; }}
    .meta {{ color: #6f768a; margin-bottom: 26px; }}
    .summary {{ background: #f8fafc; border-left: 4px solid #5477c4; padding: 14px 18px; }}
    figure {{ margin: 22px 0 30px; }}
    figure img {{ width: 100%; border: 1px solid #d7dbe7; }}
    figcaption {{ color: #6f768a; font-size: 13px; margin-top: 8px; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 14px 0 24px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e6e8f0; padding: 7px 8px; text-align: right; }}
    .data-table th:first-child, .data-table td:first-child,
    .data-table th:nth-child(2), .data-table td:nth-child(2) {{ text-align: left; }}
    .data-table th {{ background: #f4f5f7; color: #1f2430; }}
    .note {{ color: #6f768a; font-size: 13px; }}
    code {{ background: #f4f5f7; padding: 1px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>货币信用四象限资产表现显著性分析</h1>
  <div class="meta">数据源：国信证券货币信用划分、日涨跌幅_填充.csv；生成日期：{pd.Timestamp.today().strftime("%Y-%m-%d")}</div>

  <h2>技术摘要</h2>
  <div class="summary"><ul>{''.join(summary_items)}</ul></div>

  <h2>关键发现与视觉证据</h2>
  <p>下表按 Kruskal-Wallis 的 FDR 校正后 p 值排序。Kruskal-Wallis 是本报告主检验；Welch ANOVA 作为均值差异的补充参考。显著性判断优先看 <code>Kruskal_FDR</code>。</p>
  {make_table_html(overall[overall_cols], max_rows=12)}

  <figure>
    <img src="{html.escape(FIG_SIG)}" alt="显著性检验摘要图">
    <figcaption>三类有效象限整体差异的多重检验校正结果。竖线分别表示 FDR 5% 和 10% 阈值。</figcaption>
  </figure>

  <p>年化收益热力图展示四象限的描述性表现。紧货币宽信用只有 1 个月，其年化数字容易被单月收益放大，不能作为稳健结论。</p>
  <figure>
    <img src="{html.escape(FIG_HEATMAP)}" alt="四象限资产年化收益热力图">
    <figcaption>单元格为各资产在对应象限下按月收益复利得到的年化收益，单位为 %。</figcaption>
  </figure>

  <p>箱线图只使用样本足够的三类象限，用于观察收益分布、离群点和中位数差异。</p>
  <figure>
    <img src="{html.escape(FIG_BOX)}" alt="三类有效象限资产月收益箱线图">
    <figcaption>月收益单位为 %。图中未包含只有 1 个月样本的紧货币宽信用。</figcaption>
  </figure>

  <h2>剔除明显弱势周期后的三类资产池</h2>
  <p>资产池只在三类有效周期内构建。规则是：对每个资产分别比较三类有效周期的年化收益，若某周期较该资产最佳周期低至少 {UNDERPERFORMANCE_THRESHOLD_PP:.0f} 个百分点，且不高于该资产三周期年化收益中位数，则剔除该资产在该周期的配置资格。</p>
  {make_table_html(asset_pools, max_rows=3)}
  <figure>
    <img src="{html.escape(FIG_POOL)}" alt="三周期优选资产池">
    <figcaption>资产池用于表达相对更适合的周期配置集合，不代表显著性检验通过，也不是因果结论。</figcaption>
  </figure>

  <h3>资产-周期剔除明细</h3>
  {make_table_html(asset_pool_detail[["资产", CYCLE_COLUMN, "年化收益(%)", "该资产最佳周期", "与最佳周期差距(百分点)", "三周期年化收益中位数(%)", "是否剔除", "剔除原因"]], max_rows=36)}

  <h2>各资产表现最好的有效象限</h2>
  <p>下表只在三类有效象限内比较年化收益最高的象限，用于描述表现方向，不等同于显著性结论。</p>
  {make_table_html(top_desc, max_rows=12)}

  <h2>Pairwise 检验结果</h2>
  <p>仅对整体 Kruskal-Wallis 原始 p 值或 FDR 小于等于 10% 的资产做两两 Mann-Whitney U 检验，并再次做 FDR 校正。Cliff's delta 为组A相对组B的非参数效应量，正值表示组A整体收益更高。</p>
  {pair_section}

  <h2>范围、定义与方法</h2>
  <ul>
    <li>资产范围：排除 <code>一天期国债逆回购</code> 后的 12 个风险资产。</li>
    <li>收益单位：输入日涨跌幅为百分比点；月收益按 <code>prod(1 + 日收益/100) - 1</code> 复利后再乘以 100。</li>
    <li>主检验样本：宽货币紧信用、宽货币宽信用、紧货币紧信用三类。紧货币宽信用只有 1 个月，仅做描述。</li>
    <li>整体差异：Kruskal-Wallis 非参数检验为主；Welch ANOVA 为补充。</li>
    <li>多重检验：整体检验和 pairwise 检验分别使用 Benjamini-Hochberg FDR 校正。</li>
    <li>资产池剔除规则：某资产某周期年化收益较该资产最佳周期低至少 {UNDERPERFORMANCE_THRESHOLD_PP:.0f} 个百分点，且不高于该资产三周期年化收益中位数。</li>
  </ul>

  <h2>限制、稳健性与后续建议</h2>
  <ul>
    <li>象限样本高度不均衡，尤其紧货币紧信用只有 14 个月、紧货币宽信用只有 1 个月；结论适合作为历史条件相关性证据，不应解读为因果关系。</li>
    <li>本报告使用填充版收益数据，优点是资产历史连续，缺点是早期部分资产可能包含代理填充口径。</li>
    <li>后续若用于资产配置规则，建议进一步加入未填充交易收益口径、滚动窗口检验和样本外检验。</li>
  </ul>

  <p class="note">配套明细文件：象限月度收益明细.csv、象限描述统计.csv、显著性检验结果.csv、资产周期剔除明细.csv、三周期优选资产池.csv。</p>
</main>
</body>
</html>
"""
    (OUTPUT_DIR / "report.html").write_text(html_text, encoding="utf-8")


def validate_outputs(panel: pd.DataFrame, long_panel: pd.DataFrame, assets: list[str]) -> dict[str, object]:
    counts = panel[CYCLE_COLUMN].astype(str).value_counts().to_dict()
    missing_assets = long_panel.groupby("资产")[RETURN_COLUMN].apply(lambda s: int(s.isna().sum())).to_dict()
    expected_files = [
        "report.html",
        "象限月度收益明细.csv",
        "象限描述统计.csv",
        "显著性检验结果.csv",
        "资产周期剔除明细.csv",
        "三周期优选资产池.csv",
        FIG_BOX,
        FIG_HEATMAP,
        FIG_SIG,
        FIG_POOL,
    ]
    file_sizes = {name: (OUTPUT_DIR / name).stat().st_size for name in expected_files if (OUTPUT_DIR / name).exists()}

    checks = {
        "merged_months": int(panel[MONTH_COLUMN].nunique()),
        "asset_count": len(assets),
        "cycle_counts": {k: int(v) for k, v in counts.items()},
        "missing_monthly_returns_by_asset": missing_assets,
        "output_file_sizes": file_sizes,
    }

    if checks["merged_months"] != 132:
        raise AssertionError(f"合并后月份数应为 132，实际为 {checks['merged_months']}")
    if len(assets) != 12:
        raise AssertionError(f"风险资产数应为 12，实际为 {len(assets)}")
    if any(v != 0 for v in missing_assets.values()):
        raise AssertionError(f"月收益存在缺失: {missing_assets}")
    for cycle, expected in EXPECTED_CYCLE_COUNTS.items():
        actual = counts.get(cycle, 0)
        if actual != expected:
            raise AssertionError(f"{cycle} 样本月数应为 {expected}，实际为 {actual}")
    missing_files = sorted(set(expected_files) - set(file_sizes))
    if missing_files:
        raise AssertionError(f"缺少输出文件: {missing_files}")
    empty_files = {name: size for name, size in file_sizes.items() if size <= 0}
    if empty_files:
        raise AssertionError(f"存在空输出文件: {empty_files}")
    return checks


def main() -> None:
    configure_plot_style()
    ensure_output_dir()

    cycle = read_cycle_data()
    daily, assets = read_daily_returns()
    monthly = compound_monthly_returns(daily, assets)
    panel = build_monthly_panel(cycle, monthly)
    long_panel = to_long_panel(panel, assets)
    description = describe_by_cycle(long_panel)
    tests = run_significance_tests(long_panel, assets)
    asset_pool_detail, asset_pools = classify_asset_cycle_pools(description)

    save_csvs(long_panel, description, tests, asset_pool_detail, asset_pools)
    save_boxplot(long_panel, assets)
    save_heatmap(description, assets)
    save_significance_chart(tests)
    save_asset_pool_chart(asset_pools)
    build_report(panel, long_panel, description, tests, asset_pool_detail, asset_pools, assets)

    checks = validate_outputs(panel, long_panel, assets)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
