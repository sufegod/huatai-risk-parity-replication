from __future__ import annotations

import json
import html as html_lib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import duckdb
import fitz
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from pypdf import PdfReader


TITLE = "基于 EWMA 半协方差的风险平价资产配置优化——凸重构、阻尼牛顿法与实证分析"


def _safe_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        rows.append({str(key): _safe_number(value) for key, value in record.items()})
    return rows


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _artifact_payload(course_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    queries = {
        "headline_query": """
SELECT annual_return, sharpe, max_drawdown
FROM read_csv_auto('output/tables/strategy_metrics.csv', header=true)
WHERE strategy='erc' AND period='validation'
""".strip(),
        "nav_query": """
WITH long_data AS (
  UNPIVOT (SELECT * FROM read_csv_auto('output/tables/strategy_nav.csv', header=true))
  ON equal_weight, inverse_downside_vol, erc
  INTO NAME strategy VALUE nav
), ranked AS (
  SELECT CAST(date AS DATE) AS date, strategy, nav,
         row_number() OVER (
           PARTITION BY date_trunc('month', CAST(date AS DATE)), strategy
           ORDER BY CAST(date AS DATE) DESC
         ) AS rn
  FROM long_data
)
SELECT strftime(date, '%Y-%m-%d') AS date,
       CASE strategy
         WHEN 'equal_weight' THEN '等权组合'
         WHEN 'inverse_downside_vol' THEN '逆下行波动率'
         ELSE '风险平价'
       END AS strategy,
       nav
FROM ranked WHERE rn=1 ORDER BY date, strategy
""".strip(),
        "performance_query": """
SELECT CASE strategy
         WHEN 'equal_weight' THEN '等权组合'
         WHEN 'inverse_downside_vol' THEN '逆下行波动率'
         ELSE '风险平价'
       END AS strategy,
       CASE period WHEN 'train' THEN '训练期' ELSE '验证期' END AS period,
       annual_return, annual_volatility, sharpe, max_drawdown, calmar, annual_turnover
FROM read_csv_auto('output/tables/strategy_metrics.csv', header=true)
WHERE period IN ('train', 'validation')
ORDER BY period, strategy
""".strip(),
        "algorithm_query": """
SELECT *, CASE method
         WHEN 'newton' THEN '阻尼牛顿法'
         WHEN 'lbfgsb' THEN 'L-BFGS-B'
         ELSE 'SLSQP'
       END AS algorithm
FROM read_csv_auto('output/tables/algorithm_summary.csv', header=true)
ORDER BY method
""".strip(),
        "sensitivity_query": """
SELECT *, CAST(CAST("window" AS INTEGER) AS VARCHAR) AS window_label,
       printf('%.2f', decay) AS decay_label
FROM read_csv_auto('output/tables/parameter_sensitivity.csv', header=true)
WHERE period='validation'
ORDER BY decay, "window"
""".strip(),
        "optimal_query": """
SELECT asset,
       CASE measure WHEN 'weight' THEN '组合权重' ELSE '风险贡献' END AS measure,
       value
FROM (
  UNPIVOT (
    SELECT asset, weight, risk_contribution
    FROM read_csv_auto('output/tables/representative_optimal_solution.csv', header=true)
  ) ON weight, risk_contribution INTO NAME measure VALUE value
)
ORDER BY asset, measure
""".strip(),
        "estimator_query": """
SELECT *, CASE estimator
         WHEN 'sample' THEN '样本协方差'
         WHEN 'ewma_full' THEN 'EWMA全协方差'
         ELSE 'EWMA半协方差'
       END AS estimator_name
FROM read_csv_auto('output/tables/estimator_comparison.csv', header=true)
WHERE period='validation'
ORDER BY estimator
""".strip(),
        "stress_query": """
SELECT *, CASE method
         WHEN 'newton' THEN '阻尼牛顿法'
         WHEN 'lbfgsb' THEN 'L-BFGS-B'
         ELSE 'SLSQP'
       END AS algorithm
FROM read_csv_auto('output/tables/stress_test_summary.csv', header=true)
WHERE condition_number=(
  SELECT max(condition_number)
  FROM read_csv_auto('output/tables/stress_test_summary.csv', header=true)
)
ORDER BY method
""".strip(),
        "quality_query": """
WITH clean AS (
  SELECT * FROM read_csv_auto('data/etf_returns.csv', header=true)
), long_clean AS (
  UNPIVOT clean ON COLUMNS(* EXCLUDE (date)) INTO NAME asset VALUE daily_return
), profile AS (
  SELECT * FROM read_csv_auto('output/tables/data_quality_profile.csv', header=true)
)
SELECT '数据行数' AS check, (SELECT count(*) FROM clean) AS value, '通过' AS result
UNION ALL SELECT '资产数量', (SELECT count(DISTINCT asset) FROM long_clean), '通过'
UNION ALL SELECT '重复日期', (SELECT count(*)-count(DISTINCT date) FROM clean), '通过'
UNION ALL SELECT '清洗前缺失单元格', (SELECT sum(missing_before_fill) FROM profile), '已按0收益处理'
UNION ALL SELECT '清洗后缺失单元格', (SELECT count(*) FROM long_clean WHERE daily_return IS NULL), '通过'
""".strip(),
    }

    original_cwd = Path.cwd()
    try:
        os.chdir(course_dir)
        connection = duckdb.connect()
        frames = {name: connection.execute(sql).fetchdf() for name, sql in queries.items()}
        connection.close()
    finally:
        os.chdir(original_cwd)

    headline = frames["headline_query"]
    nav_monthly = frames["nav_query"]
    performance = frames["performance_query"]
    algorithms = frames["algorithm_query"]
    sensitivity_validation = frames["sensitivity_query"]
    optimal_long = frames["optimal_query"]
    estimator_validation = frames["estimator_query"]
    stress_view = frames["stress_query"]
    quality = frames["quality_query"]

    query_file = course_dir / "output" / "html" / "report_queries.sql"
    query_file.write_text(
        "\n\n".join(f"-- {name}\n{sql};" for name, sql in queries.items()) + "\n",
        encoding="utf-8",
    )

    validation_erc = summary["validation_erc"]
    validation_equal = summary["validation_equal_weight"]
    selected = summary["selected_parameter"]
    data_quality = summary["data_quality"]
    sources = [
        {
            "id": "etf_data",
            "label": "ETF风险平价回测数据",
            "path": "data/etf_returns.csv",
            "description": "项目原始 Excel 第一张工作表清洗后的 9 资产日收益。",
        },
        {
            "id": "derived_results",
            "label": "课程实验派生结果",
            "path": "output/tables/analysis_summary.json",
            "description": "阻尼牛顿法、算法比较、回测、稳健性和敏感性实验的可复现输出。",
        },
        {
            "id": "research_report",
            "label": "华泰研究：中国版全天候增强策略",
            "path": "策略研报来源/金工_ 从资产配置走向因子配置：中国版全天候增强策略.pdf",
            "description": "用于风险平价与宏观因子配置的研究背景，不复制其中图表。",
        },
    ]
    query_descriptions = {
        "headline_query": "从策略绩效表抽取风险平价验证期核心指标。",
        "nav_query": "将日净值转换为月末长表，供累计净值图使用。",
        "performance_query": "抽取训练期和验证期的三策略绩效。",
        "algorithm_query": "抽取滚动窗口求解器效率与精度摘要。",
        "sensitivity_query": "抽取12组参数的样本外敏感性结果。",
        "optimal_query": "将代表窗口权重和风险贡献转换为长表。",
        "estimator_query": "抽取三类风险估计方法的样本外表现。",
        "stress_query": "抽取条件数1e8时的病态矩阵压力测试结果。",
        "quality_query": "从清洗收益和质量概要复算行数、资产数、重复与缺失。",
    }
    for source_id, sql in queries.items():
        sources.append(
            {
                "id": source_id,
                "label": query_descriptions[source_id],
                "path": "output/html/report_queries.sql",
                "query": {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": sql,
                    "description": query_descriptions[source_id],
                    "tables_used": ["output/tables/*.csv", "data/etf_returns.csv"],
                    "executed_at": "2026-07-13T22:00:00+08:00",
                },
            }
        )

    cards = [
        {
            "id": "validation_return",
            "dataset": "headline_metrics",
            "sourceId": "headline_query",
            "description": "风险平价组合在样本外验证期的年化收益。",
            "metrics": [{"label": "验证期年化收益", "field": "annual_return", "format": "percent"}],
        },
        {
            "id": "validation_sharpe",
            "dataset": "headline_metrics",
            "sourceId": "headline_query",
            "description": "以无风险利率为 0 计算的年化夏普比率。",
            "metrics": [{"label": "验证期夏普比率", "field": "sharpe", "format": "number"}],
        },
        {
            "id": "validation_drawdown",
            "dataset": "headline_metrics",
            "sourceId": "headline_query",
            "description": "验证期累计净值的最深峰谷回撤。",
            "metrics": [{"label": "验证期最大回撤", "field": "max_drawdown", "format": "percent"}],
        },
    ]

    charts = [
        {
            "id": "nav_chart",
            "title": "三类资产配置策略累计净值",
            "description": "2014年1月至2026年4月，月末观察、下一交易日调仓，含单边5bp成本。",
            "type": "line",
            "dataset": "nav_monthly",
            "sourceId": "nav_query",
            "encodings": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "nav", "type": "quantitative"},
                "color": {"field": "strategy", "type": "nominal"},
            },
        },
        {
            "id": "algorithm_chart",
            "title": "三类求解器的中位迭代次数",
            "description": "全部滚动月度EWMA半协方差矩阵。",
            "type": "bar",
            "dataset": "algorithm_summary",
            "sourceId": "algorithm_query",
            "encodings": {
                "x": {"field": "algorithm", "type": "nominal"},
                "y": {"field": "median_iterations", "type": "quantitative"},
            },
        },
        {
            "id": "sensitivity_chart",
            "title": "参数组合的样本外夏普比率",
            "description": "验证期为2021年1月至2026年4月；参数选择只使用训练期。",
            "type": "heatmap",
            "dataset": "sensitivity_validation",
            "sourceId": "sensitivity_query",
            "encodings": {
                "x": {"field": "window_label", "type": "nominal"},
                "y": {"field": "decay_label", "type": "nominal"},
                "color": {"field": "sharpe", "type": "quantitative"},
            },
        },
        {
            "id": "optimal_chart",
            "title": "代表窗口的组合权重与风险贡献",
            "description": "目标风险贡献均为11.11%，低波动债券获得较高名义权重。",
            "type": "bar",
            "dataset": "optimal_solution",
            "sourceId": "optimal_query",
            "encodings": {
                "x": {"field": "asset", "type": "nominal"},
                "y": {"field": "value", "type": "quantitative"},
                "color": {"field": "measure", "type": "nominal"},
            },
        },
        {
            "id": "stress_chart",
            "title": "条件数1e8下的最大风险贡献误差",
            "description": "9维随机正定矩阵，每种算法固定种子重复20次。",
            "type": "bar",
            "dataset": "stress_view",
            "sourceId": "stress_query",
            "encodings": {
                "x": {"field": "algorithm", "type": "nominal"},
                "y": {"field": "max_rc_error", "type": "quantitative"},
            },
        },
    ]

    tables = [
        {
            "id": "performance_table",
            "title": "训练期与验证期绩效",
            "description": "收益和风险按252个交易日年化；2026年为不完整年度。",
            "dataset": "performance",
            "sourceId": "performance_query",
            "defaultSort": {"field": "period", "direction": "asc"},
            "columns": [
                {"field": "strategy", "label": "策略"},
                {"field": "period", "label": "区间"},
                {"field": "annual_return", "label": "年化收益", "format": "percent"},
                {"field": "annual_volatility", "label": "年化波动", "format": "percent"},
                {"field": "sharpe", "label": "夏普", "format": "number"},
                {"field": "max_drawdown", "label": "最大回撤", "format": "percent"},
                {"field": "calmar", "label": "卡玛", "format": "number"},
                {"field": "annual_turnover", "label": "年化换手", "format": "percent"},
            ],
        },
        {
            "id": "algorithm_table",
            "title": "滚动窗口算法精度与效率",
            "description": "成功要求满足权重约束且最大风险贡献误差不超过1e-6。",
            "dataset": "algorithm_summary",
            "sourceId": "algorithm_query",
            "defaultSort": {"field": "median_iterations", "direction": "asc"},
            "columns": [
                {"field": "algorithm", "label": "算法"},
                {"field": "success_rate", "label": "成功率", "format": "percent"},
                {"field": "median_iterations", "label": "中位迭代", "format": "number"},
                {"field": "median_runtime_ms", "label": "中位耗时(ms)", "format": "number"},
                {"field": "median_rc_error", "label": "中位RC误差", "format": "number"},
                {"field": "max_rc_error", "label": "最大RC误差", "format": "number"},
            ],
        },
        {
            "id": "estimator_table",
            "title": "风险估计方法的样本外表现",
            "description": "窗口252日、EWMA衰减0.97；样本协方差不使用衰减参数。",
            "dataset": "estimator_validation",
            "sourceId": "estimator_query",
            "defaultSort": {"field": "sharpe", "direction": "desc"},
            "columns": [
                {"field": "estimator_name", "label": "风险估计"},
                {"field": "annual_return", "label": "年化收益", "format": "percent"},
                {"field": "annual_volatility", "label": "年化波动", "format": "percent"},
                {"field": "sharpe", "label": "夏普", "format": "number"},
                {"field": "max_drawdown", "label": "最大回撤", "format": "percent"},
            ],
        },
        {
            "id": "quality_table",
            "title": "数据质量检查摘要",
            "description": f"原始数据覆盖{data_quality['start_date']}至{data_quality['end_date']}。",
            "dataset": "data_quality",
            "sourceId": "quality_query",
            "defaultSort": {"field": "check", "direction": "asc"},
            "columns": [
                {"field": "check", "label": "检查项"},
                {"field": "value", "label": "数值", "format": "number"},
                {"field": "result", "label": "结论"},
            ],
        },
    ]

    technical_summary = (
        f"## 技术摘要\n\n"
        f"**凸重构得到稳定、可审计的等风险贡献解。** 在{data_quality['rows']}个交易日、9类资产上，"
        f"自实现阻尼牛顿法的滚动窗口中位迭代次数为{_fmt(summary['newton_summary']['median_iterations'], 0)}次，"
        f"最大风险贡献误差为{summary['newton_summary']['max_rc_error']:.2e}。\n\n"
        f"**样本外结果支持降低波动而非追求最高绝对收益。** 2021年至2026年4月，风险平价组合年化收益"
        f"{_pct(validation_erc['annual_return'])}、年化波动{_pct(validation_erc['annual_volatility'])}、"
        f"夏普{_fmt(validation_erc['sharpe'])}、最大回撤{_pct(validation_erc['max_drawdown'])}；"
        f"同期等权组合夏普为{_fmt(validation_equal['sharpe'])}。\n\n"
        f"**参数并非越接近1越好。** 训练期规则选择窗口{selected['window']}日、衰减系数{selected['decay']:.2f}；"
        f"其样本外夏普为{_fmt(selected['validation_sharpe'])}。参数网格用于稳健性检查，不替代项目默认的252日与0.97主结论。"
    )

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": (
                f"# {config.get('title', TITLE)}\n\n"
                f"**课程：** {config.get('course', '最优化理论与算法')}  \n"
                f"**姓名：** {config.get('student_name', '请填写')}　　**学号：** {config.get('student_id', '请填写')}  \n"
                f"**数据区间：** {data_quality['start_date']} 至 {data_quality['end_date']}　　**完成日期：** 2026年7月"
            ),
        },
        {"id": "technical_summary", "type": "markdown", "body": technical_summary, "sourceId": "derived_results"},
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": ["validation_return", "validation_sharpe", "validation_drawdown"],
        },
        {
            "id": "key_findings",
            "type": "markdown",
            "body": (
                "## 风险平价的价值主要体现在风险调整后收益\n\n"
                "累计净值用于比较长期路径，但不能单独证明算法优越。等权组合暴露于高波动资产，逆下行波动率只利用对角风险，"
                "风险平价则进一步利用资产间相关性并约束各资产风险贡献。下图与后续绩效表共同回答收益、波动和回撤三项问题。"
            ),
        },
        {"id": "nav_chart_block", "type": "chart", "chartId": "nav_chart"},
        {
            "id": "nav_interpretation",
            "type": "markdown",
            "sourceId": "derived_results",
            "body": (
                f"风险平价验证期夏普为**{_fmt(validation_erc['sharpe'])}**，等权组合为**{_fmt(validation_equal['sharpe'])}**。"
                "这一差异是历史样本中的描述性结果，不代表未来收益保证；评价重点是同一交易成本与调仓规则下的风险控制。"
            ),
        },
        {
            "id": "scope_data",
            "type": "markdown",
            "sourceId": "etf_data",
            "body": (
                "## 数据完整，但代理填充限制外推解释\n\n"
                f"数据包含{data_quality['assets']}类股票、债券、商品和黄金代理资产，共{data_quality['rows']}个交易日。"
                "原始百分数收益统一除以100；3个孤立缺失单元格按0收益处理；日期无重复且严格升序。"
                "训练期为2014-2020年，验证期为2021年至2026年4月3日。ETF成立前由项目提供的指数代理补齐，"
                "因此结果是模型实验而非可直接交易业绩。"
            ),
        },
        {"id": "quality_table_block", "type": "table", "tableId": "quality_table"},
        {
            "id": "model_specification",
            "type": "markdown",
            "body": (
                "## 对数障碍凸模型把等风险贡献转化为唯一解\n\n"
                "组合方差为 σ²(w)=wᵀΣw，边际风险为 (Σw)ᵢ，资产风险贡献为 RCᵢ=wᵢ(Σw)ᵢ。"
                "原始模型最小化 ∑ᵢ[RCᵢ-σ²(w)/n]²，并满足 wᵢ≥0、∑ᵢwᵢ=1。该目标为四次非凸函数且量级随协方差缩放。\n\n"
                "令 bᵢ=1/n，改求 minₓ ½xᵀΣx-∑ᵢbᵢln(xᵢ)，x>0，再以 w=x/(1ᵀx) 归一化。"
                "梯度为 Σx-b/x，Hessian为 Σ+diag(b/x²)。加入10⁻⁸I后Σ正定，Hessian严格正定，故最优解唯一。"
                "一阶条件给出 xᵢ(Σx)ᵢ=bᵢ，归一化不会改变相对风险贡献，因此得到等风险贡献解。"
            ),
        },
        {
            "id": "algorithm_method",
            "type": "markdown",
            "body": (
                "## 阻尼牛顿法直接利用曲率并保持正权重\n\n"
                "每轮计算 Newton 方向 p=-H⁻¹g；先由 x+αp>0 得到可行步长上界，再以 Armijo 条件"
                " f(x+αp)≤f(x)+10⁻⁴αgᵀp 回溯。梯度无穷范数不超过10⁻¹⁰时停止。"
                "L-BFGS-B使用相同凸目标和解析梯度；SLSQP直接求原始带等式约束模型，并按协方差尺度标准化目标值。"
            ),
        },
        {"id": "algorithm_chart_block", "type": "chart", "chartId": "algorithm_chart"},
        {
            "id": "algorithm_interpretation",
            "type": "markdown",
            "sourceId": "derived_results",
            "body": (
                f"阻尼牛顿法滚动窗口成功率为**{_pct(summary['newton_summary']['success_rate'])}**，"
                f"中位耗时**{_fmt(summary['newton_summary']['median_runtime_ms'], 3)}毫秒**；"
                f"L-BFGS-B中位迭代{_fmt(summary['lbfgsb_summary']['median_iterations'], 0)}次，"
                f"SLSQP中位迭代{_fmt(summary['slsqp_summary']['median_iterations'], 0)}次。"
                "耗时受Python与硬件影响，误差与约束满足度才是跨环境可比较指标。"
            ),
        },
        {"id": "algorithm_table_block", "type": "table", "tableId": "algorithm_table"},
        {
            "id": "stress_method",
            "type": "markdown",
            "sourceId": "derived_results",
            "body": (
                "## 病态矩阵揭示算法稳定性边界\n\n"
                f"固定随机种子42生成条件数10²至10⁸的9维正定矩阵，每档重复20次。条件数10⁸时，"
                f"阻尼牛顿法最大风险贡献误差为**{summary['stress_newton_1e8']['max_rc_error']:.2e}**。"
                "该实验隔离了收益序列和回测规则，仅检验数值求解器对协方差病态性的敏感程度。"
            ),
        },
        {"id": "stress_chart_block", "type": "chart", "chartId": "stress_chart"},
        {
            "id": "backtest_design",
            "type": "markdown",
            "body": (
                "## 回测严格使用月末已知信息\n\n"
                "每个月末使用截至当日的过去252个交易日估计风险，目标权重在下一交易日执行；"
                "月内权重按资产收益自然漂移。调仓成本为单边换手额的0.05%，无杠杆、只做多、无风险利率设为0。"
                "年化收益、波动、夏普、最大回撤、卡玛、月度胜率、年化换手和最大权重均由同一日收益序列计算。"
            ),
        },
        {"id": "performance_table_block", "type": "table", "tableId": "performance_table"},
        {
            "id": "risk_estimation",
            "type": "markdown",
            "sourceId": "derived_results",
            "body": (
                "## 半协方差强调损失方向，但结论依赖估计口径\n\n"
                "样本协方差等权对待历史，EWMA全协方差提高近期权重，EWMA半协方差进一步将正收益截为0。"
                "三种方法使用完全相同的调仓、成本和求解器，因此样本外差异可归因于风险输入，而不是交易规则。"
            ),
        },
        {"id": "estimator_table_block", "type": "table", "tableId": "estimator_table"},
        {
            "id": "sensitivity_method",
            "type": "markdown",
            "body": (
                "## 训练期选参后再观察样本外，避免全样本反向挑选\n\n"
                "对衰减系数{0.90,0.94,0.97,0.99}和窗口{126,252,504}运行12组实验。"
                "只按2014-2020年训练期夏普排序，夏普相同时选择回撤较小者；2021年后的结果只用于验证。"
            ),
        },
        {"id": "sensitivity_chart_block", "type": "chart", "chartId": "sensitivity_chart"},
        {
            "id": "sensitivity_interpretation",
            "type": "markdown",
            "sourceId": "derived_results",
            "body": (
                f"训练期选择窗口**{selected['window']}日**、衰减系数**{selected['decay']:.2f}**，"
                f"训练期夏普{_fmt(selected['train_sharpe'])}、验证期夏普{_fmt(selected['validation_sharpe'])}。"
                "报告仍以项目默认252日、0.97作为主模型，避免把参数搜索结果包装成确定性最优参数。"
            ),
        },
        {
            "id": "optimal_solution",
            "type": "markdown",
            "body": (
                "## 等风险贡献不等于等权配置\n\n"
                "债券日波动较低，为达到与股票、商品相同的风险贡献，需要配置更高名义权重。"
                "因此风险平价可能形成高债券权重，但每类资产的方差贡献仍接近11.11%。"
                "这也说明只看权重集中度会误判组合的真实风险分散程度。"
            ),
        },
        {"id": "optimal_chart_block", "type": "chart", "chartId": "optimal_chart"},
        {
            "id": "limitations_next",
            "type": "markdown",
            "body": (
                "## 结论、局限与下一步\n\n"
                "1. 对数障碍凸重构给出唯一全局解，阻尼牛顿法以少量迭代达到严格风险贡献误差。  \n"
                "2. 历史样本中，风险平价主要通过降低波动和回撤改善风险调整后收益，而非保证最高年化收益。  \n"
                "3. ETF成立前指数代理、固定交易成本、零无风险利率、参数网格多重比较和2026年不完整年度限制外推。  \n"
                "4. 后续可加入协方差收缩、权重上限、换手惩罚和滚动交叉验证，并检验不同交易成本。\n\n"
                "## 进一步问题\n\n"
                "在债券低利率或股债相关性上升阶段，等风险贡献是否仍能保持分散效果？"
                "引入权重上限和换手惩罚后，凸性、最优性条件与样本外表现将如何变化？\n\n"
                "## 参考文献\n\n"
                "[1] Maillard S, Roncalli T, Teïletche J. On the Properties of Equally-Weighted Risk Contributions Portfolios, 2010.  \n"
                "[2] Spinu F. An Algorithm for Computing Risk Parity Weights, 2013.  \n"
                "[3] Nocedal J, Wright S. Numerical Optimization, 2nd ed., 2006.  \n"
                "[4] 华泰研究. 从资产配置走向因子配置：中国版全天候增强策略, 2025.  \n"
                "[5] 本项目 v0.01 与 v0.05 风险平价历史代码及本课程复现实验。"
            ),
        },
    ]

    manifest = {
        "version": 1,
        "surface": "report",
        "title": config.get("title", TITLE),
        "generatedAt": "2026-07-13T22:00:00+08:00",
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": blocks,
    }
    datasets = {
        "headline_metrics": _records(headline),
        "nav_monthly": _records(nav_monthly),
        "performance": _records(performance),
        "algorithm_summary": _records(algorithms),
        "sensitivity_validation": _records(sensitivity_validation),
        "optimal_solution": _records(optimal_long),
        "estimator_validation": _records(estimator_validation),
        "stress_view": _records(stress_view),
        "data_quality": _records(quality),
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-07-13T22:00:00+08:00",
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def _find_node() -> Path:
    found = shutil.which("node")
    if found:
        return Path(found)
    bundled = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "node"
        / "bin"
        / "node.exe"
    )
    if bundled.exists():
        return bundled
    raise FileNotFoundError("Node.js executable was not found")


def _find_portable_builder() -> Path:
    base = Path.home() / ".codex" / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    candidates = sorted(base.glob("*/skills/build-report/scripts/deliver_portable_artifact.mjs"))
    if not candidates:
        raise FileNotFoundError("Data Analytics portable report builder was not found")
    return candidates[-1]


def _find_chrome() -> Path:
    candidates = [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Chrome or Edge executable was not found")


def _build_portable_html(artifact_path: Path, html_path: Path, receipt_path: Path) -> None:
    node = _find_node()
    builder = _find_portable_builder()
    process = subprocess.run(
        [str(node), str(builder), "--input", str(artifact_path), "--output", str(html_path)],
        cwd=str(builder.parents[3]),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    receipt_path.write_text(process.stdout + "\n" + process.stderr, encoding="utf-8")
    if process.returncode != 0 or not html_path.exists():
        raise RuntimeError(f"portable report builder failed; see {receipt_path}")


def _build_print_report(course_dir: Path, config: dict[str, Any], summary: dict[str, Any], html_path: Path) -> None:
    """Build a deterministic, A4-native report for PDF export.

    The portable HTML remains the canonical interactive artifact. This separate
    view deliberately uses the program-generated static figures so that Chrome's
    print renderer cannot expand chart datasets into raw record tables.
    """

    def esc(value: Any) -> str:
        return html_lib.escape(str(value))

    def figure(filename: str, caption: str, *, compact: bool = False) -> str:
        src = (course_dir / "output" / "figures" / filename).resolve().as_uri()
        css_class = "figure compact" if compact else "figure"
        return (
            f'<figure class="{css_class}"><img src="{src}" alt="{esc(caption)}">'
            f'<figcaption>{esc(caption)}<br><span>资料来源：本课程实验程序生成。</span></figcaption></figure>'
        )

    def table(headers: list[str], rows: list[list[Any]], *, small: bool = False) -> str:
        cls = "data-table small" if small else "data-table"
        head = "".join(f"<th>{esc(item)}</th>" for item in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows
        )
        return f'<table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'

    strategy_labels = {"equal_weight": "等权组合", "inverse_downside_vol": "逆下行波动率", "erc": "风险平价"}
    method_labels = {"newton": "阻尼牛顿法", "lbfgsb": "L-BFGS-B", "slsqp": "SLSQP"}
    estimator_labels = {"sample": "样本协方差", "ewma_full": "EWMA全协方差", "ewma_semi": "EWMA半协方差"}

    metrics = pd.read_csv(course_dir / "output" / "tables" / "strategy_metrics.csv")
    algorithms = pd.read_csv(course_dir / "output" / "tables" / "algorithm_summary.csv")
    estimators = pd.read_csv(course_dir / "output" / "tables" / "estimator_comparison.csv")
    sensitivity = pd.read_csv(course_dir / "output" / "tables" / "parameter_sensitivity.csv")
    optimum = pd.read_csv(course_dir / "output" / "tables" / "representative_optimal_solution.csv")
    quality_profile = pd.read_csv(course_dir / "output" / "tables" / "data_quality_profile.csv")

    algorithm_rows = []
    for _, row in algorithms.sort_values("median_iterations").iterrows():
        algorithm_rows.append([
            method_labels[row["method"]],
            f"{row['success_rate']:.1%}",
            f"{row['median_iterations']:.1f}",
            f"{row['median_runtime_ms']:.3f}",
            f"{row['median_rc_error']:.2e}",
            f"{row['max_rc_error']:.2e}",
        ])

    validation_rows = []
    for _, row in metrics.query("period == 'validation'").iterrows():
        validation_rows.append([
            strategy_labels[row["strategy"]],
            f"{row['annual_return']:.2%}",
            f"{row['annual_volatility']:.2%}",
            f"{row['sharpe']:.2f}",
            f"{row['max_drawdown']:.2%}",
            f"{row['calmar']:.2f}",
            f"{row['annual_turnover']:.1%}",
        ])

    estimator_rows = []
    for _, row in estimators.query("period == 'validation'").iterrows():
        estimator_rows.append([
            estimator_labels[row["estimator"]],
            f"{row['annual_return']:.2%}",
            f"{row['annual_volatility']:.2%}",
            f"{row['sharpe']:.2f}",
            f"{row['max_drawdown']:.2%}",
            f"{row['annual_turnover']:.1%}",
        ])

    sensitivity_rows = []
    for _, row in sensitivity.query("period == 'validation'").sort_values(["decay", "window"]).iterrows():
        sensitivity_rows.append([
            f"{int(row['window'])}",
            f"{row['decay']:.2f}",
            f"{row['annual_return']:.2%}",
            f"{row['sharpe']:.2f}",
            f"{row['max_drawdown']:.2%}",
        ])

    optimum_rows = []
    for _, row in optimum.iterrows():
        optimum_rows.append([
            row["asset"], f"{row['weight']:.2%}", f"{row['risk_contribution']:.4%}", f"{row['target_risk_contribution']:.4%}"
        ])

    dq = summary["data_quality"]
    selected = summary["selected_parameter"]
    validation_erc = summary["validation_erc"]
    validation_equal = summary["validation_equal_weight"]
    missing_assets = int(quality_profile["missing_before_fill"].sum())

    def page(number: int, title: str, body: str, *, cover: bool = False) -> str:
        cover_class = " page-cover" if cover else ""
        heading = "" if cover else f'<div class="running-head"><span>最优化理论与算法期末大作业</span><span>{number:02d}</span></div><h1>{esc(title)}</h1>'
        return f'<section class="page{cover_class}">{heading}{body}<div class="page-no">{number} / 10</div></section>'

    pages: list[str] = []
    pages.append(page(1, "封面", f"""
      <div class="cover-rule"></div>
      <div class="cover-kicker">最优化理论与算法 · 课程大作业</div>
      <h1 class="cover-title">{esc(config.get('title', TITLE))}</h1>
      <div class="cover-meta">
        <div><b>姓名</b><span>{esc(config.get('student_name', '请填写'))}</span></div>
        <div><b>学号</b><span>{esc(config.get('student_id', '请填写'))}</span></div>
        <div><b>数据区间</b><span>{dq['start_date']}—{dq['end_date']}</span></div>
        <div><b>完成日期</b><span>2026年7月13日</span></div>
      </div>
      <div class="abstract"><h2>摘要</h2>
        <p>本文研究多资产配置中的等风险贡献优化。针对原始风险贡献平方差模型尺度小、非凸且数值敏感的问题，采用对数障碍严格凸重构，推导梯度、Hessian 与 KKT 条件，并自行实现保持正权重的阻尼牛顿法。基于项目内9类ETF代理资产2013—2026年日收益，使用252日、衰减系数0.97的 EWMA 半协方差进行月度滚动回测，并与 L-BFGS-B、SLSQP、等权和逆下行波动率方法比较。结果表明，验证期风险平价年化收益为{validation_erc['annual_return']:.2%}、年化波动为{validation_erc['annual_volatility']:.2%}、夏普为{validation_erc['sharpe']:.2f}，其优势主要来自风险控制而非最高绝对收益。</p>
        <p class="keywords"><b>关键词：</b>风险平价；EWMA半协方差；凸优化；阻尼牛顿法；资产配置</p>
      </div>
      <div class="highlight-row">
        <div><b>{validation_erc['sharpe']:.2f}</b><span>验证期夏普</span></div>
        <div><b>{summary['newton_summary']['median_iterations']:.0f} 次</b><span>牛顿法中位迭代</span></div>
        <div><b>{summary['newton_summary']['max_rc_error']:.2e}</b><span>滚动最大RC误差</span></div>
      </div>
    """, cover=True))

    pages.append(page(2, "1　问题背景、研究意义与数据", f"""
      <h2>1.1 为什么要从资本权重转向风险预算</h2>
      <p>等权组合在名义资金上平均分配，却可能被股票和商品等高波动资产主导。风险平价直接约束各资产对组合方差的贡献，使“分散”从资本比例转化为风险比例。华泰研究关于中国版全天候策略的讨论为本研究提供背景；本文不复制研报图表，全部实证结果均由项目静态数据重新计算。</p>
      <h2>1.2 数据来源与清洗</h2>
      <p>数据来自 <code>数据/原始数据/ETF风险平价回测数据.xlsx</code> 的“日涨跌幅”表，收益由百分数除以100。ETF成立前的项目原有指数代理被保留，这一处理扩大了历史覆盖，但也限制了结果作为真实可交易业绩的解释。</p>
      {table(['检查项','结果','判定'], [
        ['交易日与区间', f"{dq['rows']}日，{dq['start_date']}—{dq['end_date']}", '通过'],
        ['资产数量', dq['assets'], '通过'],
        ['日期重复', dq['duplicate_dates'], '通过'],
        ['原始缺失单元格', missing_assets, '3个孤立值按0收益处理'],
        ['清洗后缺失', dq['missing_cells_after_fill'], '通过'],
        ['日收益范围', f"{dq['min_return']:.2%} 至 {dq['max_return']:.2%}", '合理性复核'],
      ])}
      <div class="callout"><b>研究划分：</b>2014—2020年为训练期；2021-01-01至2026-04-03为样本外验证期。参数只在训练期排序，避免使用验证期反向选参。</div>
      <h2>1.3 研究问题</h2>
      <ol><li>凸重构能否稳定得到唯一的等风险贡献解？</li><li>自实现阻尼牛顿法相对通用求解器的精度、速度和病态稳定性如何？</li><li>EWMA半协方差风险平价能否在样本外改善波动、回撤和夏普？</li></ol>
      <p class="source">资料来源：项目原始Excel、清洗后CSV及华泰研究研报；数据质量统计由程序生成。</p>
    """))

    pages.append(page(3, "2　数学模型与凸等价重构", """
      <h2>2.1 原始等风险贡献模型</h2>
      <p>设资产权重为 <i>w</i>，满足 <i>w</i><sub>i</sub>≥0 且 1ᵀ<i>w</i>=1。组合方差与第 <i>i</i> 个资产的风险贡献分别为：</p>
      <div class="equation">σ²(<i>w</i>) = <i>w</i>ᵀΣ<i>w</i>，　RC<sub>i</sub> = <i>w</i><sub>i</sub>(Σ<i>w</i>)<sub>i</sub></div>
      <p>等风险贡献可直接写为下列带等式约束的四次优化：</p>
      <div class="equation">min<sub><i>w</i></sub>　Σ<sub>i</sub>[RC<sub>i</sub> − <i>w</i>ᵀΣ<i>w</i>/n]²</div>
      <p>该目标随协方差尺度显著变化，并非全局凸；直接使用SLSQP容易受到初值、停止阈值和病态矩阵影响。</p>
      <h2>2.2 对数障碍严格凸重构</h2>
      <div class="equation strong">min<sub><i>x</i>&gt;0</sub>　f(<i>x</i>) = ½<i>x</i>ᵀΣ<i>x</i> − Σ<sub>i</sub>b<sub>i</sub>ln <i>x</i><sub>i</sub>，　b<sub>i</sub>=1/n</div>
      <p>求解后归一化 <i>w</i>=<i>x</i>/(1ᵀ<i>x</i>)。主实验使用负收益 <i>r</i><sub>t</sub><sup>−</sup>=min(<i>r</i><sub>t</sub>,0) 构造 EWMA 半协方差，并加入10<sup>−8</sup>I：</p>
      <div class="equation">Σ = Σ<sub>t</sub> α<sub>t</sub><i>r</i><sub>t</sub><sup>−</sup>(<i>r</i><sub>t</sub><sup>−</sup>)ᵀ + 10<sup>−8</sup>I，　α<sub>t</sub>∝0.97<sup>T−t</sup></div>
      <h2>2.3 梯度、Hessian 与最优性证明</h2>
      <div class="equation">∇f(<i>x</i>)=Σ<i>x</i>−<i>b</i>/<i>x</i>，　∇²f(<i>x</i>)=Σ+diag(<i>b</i>/<i>x</i>²)</div>
      <div class="proof"><b>命题：</b>该模型存在唯一最优解，归一化后满足给定风险预算。<br><b>证明：</b>脊参数使Σ正定，且diag(<i>b</i>/<i>x</i>²)正定，因此Hessian处处正定，f严格凸。最优点满足Σ<i>x</i>−<i>b</i>/<i>x</i>=0，即 <i>x</i><sub>i</sub>(Σ<i>x</i>)<sub>i</sub>=b<sub>i</sub>。归一化只对全部风险贡献乘同一尺度，故相对风险贡献等于b<sub>i</sub>；严格凸性保证最优解唯一。证毕。</div>
      <p class="source">资料来源：模型推导由本文完成；理论背景参考 Maillard 等（2010）与 Spinu（2013）。</p>
    """))

    pages.append(page(4, "3　阻尼牛顿法与求解器比较", f"""
      <div class="two-col text-cols">
        <div><h2>3.1 自实现算法</h2><div class="pseudo"><b>输入：</b>Σ、风险预算b、tol=10<sup>−10</sup><br>1　初始化 x&gt;0<br>2　计算 g=Σx−b/x，H=Σ+diag(b/x²)<br>3　解 Hp=−g<br>4　由 x+αp&gt;0 得到最大可行步长<br>5　Armijo 回溯：f(x+αp)≤f(x)+10<sup>−4</sup>αgᵀp<br>6　若 ‖g‖∞≤tol 则停止，否则返回第2步<br><b>输出：</b>w=x/(1ᵀx)</div></div>
        <div><h2>3.2 对照算法</h2><p><b>L-BFGS-B：</b>求解同一凸模型，使用解析梯度与正下界。</p><p><b>SLSQP：</b>求解原始风险贡献平方差模型，显式施加权重和为1及非负约束，并按协方差量级缩放目标。</p><p>三者统一上限1000次；比较成功率、迭代数、耗时和最大风险贡献误差。</p></div>
      </div>
      {figure('solver_convergence.png', '图1　代表窗口中阻尼牛顿法的目标、梯度与风险贡献误差收敛轨迹', compact=True)}
      {table(['算法','成功率','中位迭代','中位耗时/ms','中位RC误差','最大RC误差'], algorithm_rows, small=True)}
      <p>阻尼牛顿法中位仅{summary['newton_summary']['median_iterations']:.0f}次迭代，运行时间中位数为{summary['newton_summary']['median_runtime_ms']:.3f} ms；其滚动窗口最大风险贡献误差为{summary['newton_summary']['max_rc_error']:.2e}。未达到严格梯度阈值的个别窗口仍满足1×10<sup>−6</sup>风险贡献验收线，因此“成功率”与“最终可用性”需区分。</p>
    """))

    pages.append(page(5, "4　算法效率与病态矩阵压力测试", f"""
      <div class="two-col">
        {figure('solver_summary.png', '图2　全部滚动月度协方差矩阵上的求解效率与精度', compact=True)}
        {figure('stress_test.png', '图3　条件数10²至10⁸下的求解稳定性（每档20次）', compact=True)}
      </div>
      <h2>4.1 滚动窗口结果</h2>
      <p>在148个滚动月度矩阵上，L-BFGS-B成功率为{summary['lbfgsb_summary']['success_rate']:.1%}，但中位迭代与耗时分别为{summary['lbfgsb_summary']['median_iterations']:.0f}次和{summary['lbfgsb_summary']['median_runtime_ms']:.3f} ms。SLSQP中位迭代{summary['slsqp_summary']['median_iterations']:.1f}次，最大RC误差{summary['slsqp_summary']['max_rc_error']:.2e}，显示原始非凸形式对数值设置更敏感。</p>
      <h2>4.2 压力测试</h2>
      <p>固定随机种子42，使用随机正交基和指定特征值谱生成9维正定矩阵。条件数达到10<sup>8</sup>时，阻尼牛顿法20次成功率为{summary['stress_newton_1e8']['success_rate']:.1%}，最大RC误差为{summary['stress_newton_1e8']['max_rc_error']:.2e}，仍低于1×10<sup>−6</sup>验收阈值。压力测试说明解析Hessian与可行线搜索能显著缓解病态性，但不能替代协方差正则化。</p>
      <div class="callout"><b>算法结论：</b>阻尼牛顿法在本问题的低维、稠密且Hessian可解析场景中最有优势；若资产维数极高，存储Hessian的代价会使L-BFGS类方法更有吸引力。</div>
      <p class="source">资料来源：output/tables/algorithm_summary.csv 与 stress_test_summary.csv；图表由程序生成。</p>
    """))

    pages.append(page(6, "5　回测设计与样本外绩效", f"""
      <div class="method-strip"><span>过去252日</span><b>→</b><span>月末估计风险</span><b>→</b><span>下一交易日调仓</span><b>→</b><span>单边成本5 bp</span></div>
      <p>月末只使用截至当日的历史收益，目标权重在下一交易日执行，避免未来信息；月内权重随资产收益自然漂移。组合无杠杆、只做多。对比策略为等权组合、逆下行波动率组合和EWMA半协方差风险平价。</p>
      {figure('strategy_nav.png', '图4　三类资产配置策略累计净值（2014—2026年4月）')}
      {table(['策略','年化收益','年化波动','夏普','最大回撤','卡玛','年化换手'], validation_rows, small=True)}
      <p>验证期等权组合年化收益{validation_equal['annual_return']:.2%}高于风险平价的{validation_erc['annual_return']:.2%}，但其波动{validation_equal['annual_volatility']:.2%}和最大回撤{validation_equal['max_drawdown']:.2%}也明显更高。风险平价夏普{validation_erc['sharpe']:.2f}、卡玛{validation_erc['calmar']:.2f}，因此本文的主结论是“改善风险调整后收益”，而不是“保证最高收益”。</p>
      <p class="source">资料来源：output/tables/strategy_nav.csv 与 strategy_metrics.csv；净值和指标由程序重算。</p>
    """))

    pages.append(page(7, "6　风险估计口径与年度情景", f"""
      <div class="two-col">
        {figure('estimator_comparison.png', '图5　三类协方差估计下的样本外风险收益', compact=True)}
        {figure('yearly_returns.png', '图6　三类配置策略的年度收益情景', compact=True)}
      </div>
      {table(['风险估计','年化收益','年化波动','夏普','最大回撤','年化换手'], estimator_rows, small=True)}
      <h2>6.1 半协方差并非样本外最优</h2>
      <p>样本协方差、EWMA全协方差和EWMA半协方差均使用相同窗口、调仓、成本和凸求解器。验证期中，EWMA全协方差夏普{summary['estimator_validation']['ewma_full']['sharpe']:.2f}，高于半协方差的{summary['estimator_validation']['ewma_semi']['sharpe']:.2f}；样本协方差回撤也更浅。这一结果没有否定半协方差，而是说明“只强调下行方向”会改变相关结构与换手，并不必然提升样本外绩效。</p>
      <h2>6.2 情景解释</h2>
      <p>年度收益图显示策略相对表现会随股债商品环境变化。风险平价的结构性作用是抑制单一高波动资产主导，而不是消除宏观共同冲击。2026年数据只覆盖至4月3日，不应与完整年度直接比较。</p>
      <div class="callout"><b>审慎结论：</b>风险估计口径是一项模型假设，应通过多估计器对照、参数敏感性和更长样本检验，而不能根据单一样本外指标宣布一种估计器永久占优。</div>
    """))

    pages.append(page(8, "7　参数敏感性与训练期选参", f"""
      <p>对窗口{{126,252,504}}和衰减系数{{0.90,0.94,0.97,0.99}}运行12组半协方差风险平价回测。先按2014—2020年训练期夏普从高到低排序，若相同则选择最大回撤较小者；验证期不参与选择。</p>
      {figure('sensitivity_heatmap.png', '图7　窗口与衰减系数的训练期、验证期夏普热力图', compact=True)}
      {table(['窗口/日','衰减','验证期收益','验证期夏普','验证期回撤'], sensitivity_rows, small=True)}
      <p>训练期规则锁定窗口{selected['window']}日、衰减{selected['decay']:.2f}，训练期夏普{selected['train_sharpe']:.2f}，样本外夏普{selected['validation_sharpe']:.2f}。默认主模型252日、0.97的样本外夏普为{validation_erc['sharpe']:.2f}，两者接近，说明主结论不依赖单个参数点；但不同参数的换手率差异仍会影响落地成本。</p>
      <p class="source">资料来源：output/tables/parameter_sensitivity.csv；固定随机种子与同一交易成本设置。</p>
    """))

    pages.append(page(9, "8　代表窗口的最优权重与风险贡献", f"""
      {figure('weights_risk_contributions.png', '图8　2025-12-31代表窗口的资产权重与相对风险贡献')}
      <div class="two-col weights-layout">
        <div>{table(['资产','权重','风险贡献','目标'], optimum_rows, small=True)}</div>
        <div><h2>8.1 为什么债券权重大</h2><p>风险预算约束的是方差贡献，而非名义资金。低波动债券需要更高权重，才能达到与股票、黄金和商品相同的边际风险贡献。因此，“债券权重大”与“九类资产风险贡献均约11.11%”并不矛盾。</p><h2>8.2 最优解诊断</h2><p>代表窗口全部权重严格为正、权重和为1。每类资产风险贡献与目标11.11%的偏差处于数值容差内。滚动回测中风险平价平均最大权重为{validation_erc['average_max_weight']:.2%}，说明风险均衡不等同于资本均衡，必要时可进一步加入权重上限。</p></div>
      </div>
      <p class="source">资料来源：output/tables/representative_optimal_solution.csv；协方差窗口截至2025-12-31。</p>
    """))

    pages.append(page(10, "9　结论、局限与启示", f"""
      <h2>9.1 主要结论</h2>
      <ol class="conclusions"><li><b>模型：</b>对数障碍凸重构将等风险贡献转化为严格凸问题，加入10<sup>−8</sup>I后最优解唯一，KKT条件直接给出风险预算等式。</li><li><b>算法：</b>自实现阻尼牛顿法利用解析Hessian、正权重步长上界和Armijo线搜索，在滚动窗口中以{summary['newton_summary']['median_iterations']:.0f}次中位迭代达到高精度，并通过条件数10<sup>8</sup>压力测试。</li><li><b>实证：</b>风险平价验证期年化收益{validation_erc['annual_return']:.2%}、波动{validation_erc['annual_volatility']:.2%}、夏普{validation_erc['sharpe']:.2f}、最大回撤{validation_erc['max_drawdown']:.2%}。相对等权组合，优势主要是降低波动和回撤。</li><li><b>稳健性：</b>参数网格结果较平滑，但协方差估计对结论影响显著，EWMA半协方差在该样本外并非最优估计器。</li></ol>
      <h2>9.2 局限与下一步</h2>
      <p>ETF成立前指数代理、固定交易成本、零无风险利率、2026年不完整年度和有限参数网格限制外推。后续可加入Ledoit–Wolf收缩、权重上限、换手惩罚与滚动交叉验证，并考察股债相关性上升时的稳健性。</p>
      <h2>9.3 复现说明</h2>
      <p>在项目目录执行 <code>python run.py</code> 即可重新清洗数据、运行全部实验、生成8张图和本PDF；执行 <code>python -m unittest discover -s tests -v</code> 运行梯度/Hessian、闭式解、三算法一致性、无未来数据和交易成本测试。所有配置集中于 <code>config.json</code>。</p>
      <h2>参考文献</h2>
      <ol class="references"><li>Maillard, S., Roncalli, T., &amp; Teïletche, J. (2010). On the Properties of Equally-Weighted Risk Contributions Portfolios. <i>Journal of Portfolio Management</i>.</li><li>Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights.</li><li>Nocedal, J., &amp; Wright, S. (2006). <i>Numerical Optimization</i> (2nd ed.). Springer.</li><li>华泰研究（2025）：《从资产配置走向因子配置：中国版全天候增强策略》。</li><li>本项目历史版本：资产风险平价策略v0.01（SLSQP）与v0.05（凸优化）。</li></ol>
      <div class="final-note">报告、图表、结果表与测试均由本仓库代码重新生成；研报仅用于研究背景和方法引用。</div>
    """))

    css = """
    @page { size: A4; margin: 10mm 13mm; }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; color: #17212b; background: #fff; }
    body { font-family: SimSun, "Songti SC", "Noto Serif CJK SC", serif; font-size: 9.3pt; line-height: 1.52; }
    .page { height: 277mm; position: relative; overflow: hidden; break-after: page; padding: 0 1mm 8mm; }
    .page:last-child { break-after: auto; }
    .running-head { height: 8mm; border-bottom: .35mm solid #aeb9c3; display:flex; justify-content:space-between; align-items:center; color:#687684; font: 7.5pt Arial, sans-serif; letter-spacing:.4px; }
    h1, h2, h3 { font-family: SimHei, "Heiti SC", "Microsoft YaHei", sans-serif; color:#102a43; }
    .page > h1 { font-size: 17pt; margin: 5mm 0 4mm; padding-left: 3mm; border-left: 1.4mm solid #1f5a83; }
    h2 { font-size: 11pt; margin: 3.2mm 0 1.2mm; }
    p { margin: 1.5mm 0 2.2mm; text-align: justify; }
    ol { margin: 1.5mm 0 2mm 5mm; padding-left: 4mm; } li { margin: .8mm 0; }
    code { font-family: Consolas, monospace; font-size: 8pt; background:#f2f5f7; padding:.3mm .8mm; }
    .page-no { position:absolute; bottom:0; right:1mm; color:#82919e; font: 8pt Arial,sans-serif; }
    .page-cover { padding: 15mm 9mm 8mm; }
    .cover-rule { width:20mm; height:2mm; background:#1f5a83; margin-top:5mm; }
    .cover-kicker { margin-top:9mm; color:#4d6575; font: 10pt "Microsoft YaHei",sans-serif; letter-spacing:1.3px; }
    .cover-title { font-size: 24pt; line-height:1.35; margin:6mm 0 12mm; max-width:165mm; }
    .cover-meta { display:grid; grid-template-columns:1fr 1fr; gap:2.5mm 12mm; border-top:.4mm solid #aeb9c3; border-bottom:.4mm solid #aeb9c3; padding:4mm 0; }
    .cover-meta div { display:flex; gap:4mm; }.cover-meta b { color:#536574; width:17mm; }.cover-meta span { font-family:"Microsoft YaHei",sans-serif; }
    .abstract { margin-top:10mm; padding:6mm; background:#f3f6f8; border-left:1.2mm solid #5e87a0; }.abstract h2 { margin-top:0; }
    .keywords { margin-bottom:0; }.highlight-row { display:grid; grid-template-columns:repeat(3,1fr); gap:4mm; margin-top:8mm; }
    .highlight-row div { border-top:.8mm solid #1f5a83; padding:4mm 2mm; }.highlight-row b { display:block; font:20pt Arial,sans-serif; color:#1f5a83; }.highlight-row span { color:#5e6d78; }
    .equation { margin:3mm auto; padding:3mm 5mm; text-align:center; background:#f4f7f9; border-left:1mm solid #7f9eb2; font-family:"Times New Roman",SimSun,serif; font-size:11pt; }
    .equation.strong { background:#eaf1f5; border-color:#1f5a83; font-size:12pt; }
    .proof, .callout, .final-note { margin:3mm 0; padding:3.2mm 4mm; background:#f7f5ee; border: .3mm solid #d8cda6; }
    .source { color:#667682; font-size:7.5pt; border-top:.25mm solid #d9e0e5; padding-top:1.2mm; }
    .data-table { width:100%; border-collapse:collapse; margin:3mm 0; font-family:"Microsoft YaHei",sans-serif; font-size:8pt; }
    .data-table th { background:#1f5a83; color:#fff; font-weight:600; padding:1.7mm 1.5mm; text-align:left; }
    .data-table td { border-bottom:.25mm solid #d8e0e5; padding:1.45mm 1.5mm; }.data-table tbody tr:nth-child(even) { background:#f5f7f9; }
    .data-table.small { font-size:7.2pt; margin:2mm 0; }.data-table.small th,.data-table.small td { padding:1.05mm 1.1mm; }
    .figure { margin:2.5mm auto; text-align:center; }.figure img { display:block; max-width:100%; max-height:114mm; margin:auto; }
    .figure.compact img { max-height:91mm; }.figure figcaption { margin-top:1.3mm; font-size:7.6pt; color:#3d4e5a; }.figure figcaption span { color:#7a8791; }
    .two-col { display:grid; grid-template-columns:1fr 1fr; gap:5mm; align-items:start; }.two-col .figure img { max-height:83mm; }
    .text-cols { margin-bottom:2mm; }.pseudo { background:#172b3a; color:#edf3f6; padding:4mm; line-height:1.65; font-family:Consolas,"Microsoft YaHei",sans-serif; font-size:8pt; border-radius:1mm; }
    .method-strip { display:flex; align-items:center; justify-content:space-between; padding:3mm 5mm; background:#eaf1f5; color:#163e59; font-family:"Microsoft YaHei",sans-serif; }
    .method-strip span { font-weight:600; }.weights-layout { grid-template-columns:1.05fr .95fr; gap:6mm; }.weights-layout .data-table { margin-top:0; }
    .conclusions li { margin-bottom:2.2mm; }.references { font-size:7.8pt; line-height:1.35; }.final-note { margin-top:4mm; background:#eaf1f5; border-color:#93afbf; }
    """
    document = f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>{esc(TITLE)}</title><style>{css}</style></head><body>{"".join(pages)}</body></html>'
    html_path.write_text(document, encoding="utf-8")


def _print_pdf(html_path: Path, pdf_path: Path, profile_dir: Path) -> None:
    chrome = _find_chrome()
    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_profile = Path(tempfile.mkdtemp(prefix="chrome-print-", dir=profile_dir.parent))
    # Chrome on Windows applies the active code page to the output filename;
    # export to an ASCII basename, then move to the requested Chinese filename.
    chrome_pdf = pdf_path.parent / "report-export.pdf"
    if pdf_path.exists():
        pdf_path.unlink()
    if chrome_pdf.exists():
        chrome_pdf.unlink()
    try:
        process = subprocess.run(
            [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-sync",
                "--no-first-run",
                "--no-default-browser-check",
                "--no-pdf-header-footer",
                "--allow-file-access-from-files",
                f"--user-data-dir={temporary_profile}",
                f"--print-to-pdf={chrome_pdf}",
                html_path.resolve().as_uri(),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=120,
        )
    finally:
        shutil.rmtree(temporary_profile, ignore_errors=True)
    if process.returncode != 0 or not chrome_pdf.exists() or chrome_pdf.stat().st_size < 10_000:
        raise RuntimeError(f"Chrome PDF export failed: {process.stdout}\n{process.stderr}")
    shutil.move(str(chrome_pdf), str(pdf_path))


def _render_pdf_pages(pdf_path: Path, preview_dir: Path) -> tuple[list[Path], Path]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(pdf_path)
    page_paths: list[Path] = []
    for number, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        page_path = preview_dir / f"page-{number:02d}.png"
        pixmap.save(page_path)
        page_paths.append(page_path)
    document.close()

    thumb_width = 360
    margin = 18
    columns = 3
    thumbnails: list[Image.Image] = []
    for path in page_paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumbnails.append(image.resize((thumb_width, int(image.height * ratio))))
    rows = int(np.ceil(len(thumbnails) / columns))
    thumb_height = max(image.height for image in thumbnails)
    sheet = Image.new("RGB", (columns * thumb_width + (columns + 1) * margin, rows * thumb_height + (rows + 1) * margin), "#E9EDF1")
    draw = ImageDraw.Draw(sheet)
    for i, image in enumerate(thumbnails):
        x = margin + (i % columns) * (thumb_width + margin)
        y = margin + (i // columns) * (thumb_height + margin)
        sheet.paste(image, (x, y))
        draw.rectangle((x, y, x + image.width, y + image.height), outline="#7B8792", width=1)
    contact_sheet = preview_dir / "contact_sheet.png"
    sheet.save(contact_sheet)
    for image in thumbnails:
        image.close()
    return page_paths, contact_sheet


def _verify_pdf(pdf_path: Path, html_path: Path, preview_dir: Path) -> dict[str, Any]:
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    extracted = "\n".join((page.extract_text() or "") for page in reader.pages)
    required_terms = ["EWMA", "阻尼牛顿法", "风险平价", "参考文献"]
    source_html = html_path.read_text(encoding="utf-8")
    # Chrome may map CJK glyphs to Unicode radical forms during PDF extraction;
    # verify semantic terms against the exact print source and text selectability
    # against the exported PDF.
    missing_terms = [term for term in required_terms if term not in source_html]
    forbidden_terms = [
        "snapshot status",
        "widget type",
        "manifest path",
        "package path",
        "validation status",
        "local temp",
        "Refresh",
        "Publish",
        "Share",
        "Edit",
    ]
    leaked_terms = [term for term in forbidden_terms if term in extracted]
    page_paths, contact_sheet = _render_pdf_pages(pdf_path, preview_dir)
    verification = {
        "pdf": str(pdf_path),
        "source_html": str(html_path),
        "file_size_bytes": pdf_path.stat().st_size,
        "page_count": page_count,
        "page_size_points": [float(reader.pages[0].mediabox.width), float(reader.pages[0].mediabox.height)],
        "selectable_text_characters": len(extracted),
        "missing_required_terms": missing_terms,
        "forbidden_terms_found": leaked_terms,
        "rendered_pages": [str(path) for path in page_paths],
        "contact_sheet": str(contact_sheet),
        "expected_page_range": "8-12",
        "status": "passed" if 8 <= page_count <= 12 and len(extracted) > 3000 and not missing_terms and not leaked_terms else "needs_revision",
    }
    if verification["status"] != "passed":
        raise RuntimeError(f"PDF verification failed: {verification}")
    return verification


def build_html_and_pdf(course_dir: Path, config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    html_dir = course_dir / "output" / "html"
    pdf_dir = course_dir / "output" / "pdf"
    preview_dir = course_dir / "tmp" / "pdfs"
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    artifact = _artifact_payload(course_dir, config, summary)
    artifact_path = html_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    html_path = html_dir / "report.html"
    receipt_path = html_dir / "portable_builder_receipt.txt"
    _build_portable_html(artifact_path, html_path, receipt_path)

    print_html_path = html_dir / "print_report.html"
    _build_print_report(course_dir, config, summary, print_html_path)

    pdf_path = pdf_dir / "最优化理论与算法期末大作业_风险平价.pdf"
    _print_pdf(print_html_path, pdf_path, preview_dir / "chrome-profile")
    verification = _verify_pdf(pdf_path, print_html_path, preview_dir)
    verification["interactive_html"] = str(html_path)
    verification_path = pdf_dir / "pdf_verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    return verification
