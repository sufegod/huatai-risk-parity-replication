# -*- coding: utf-8 -*-
"""留一法交叉验证（jackknife）：检验剔除沪铝/PTA 是否具有特异性。
以 v0.20 的 6 个非股指资产为基线，逐一剔除 + 加回沪铝/PTA，跑同口径回测。
复用 daily_update_strategy.build_backtest_result()，临时变体脚本隔离、跑完即删。"""
import importlib.util
import os
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DUS_PATH = HERE / "daily_update_strategy.py"
V20_PATH = REPO / "策略复现与回测" / "策略代码" / "资产风险平价策略0.20（剔除沪铝PTA）.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


dus = load_module("dus_mod", DUS_PATH)
v20_src = V20_PATH.read_text(encoding="utf-8")

base_strat = dus.load_strategy_module(V20_PATH)
base_classes = {k: list(v) for k, v in base_strat.RISK_PARITY_ASSET_CLASSES.items()}
all_rp = [a for lst in base_classes.values() for a in lst]
STRAT_NAME = base_strat.STRATEGY_NAME


def make_block(classes):
    lines = ["RISK_PARITY_ASSET_CLASSES = {"]
    for k, v in classes.items():
        if v:
            lines.append(f"    '{k}': {v},")
    lines.append("}")
    return "\n".join(lines)


def variant_src(classes):
    return re.sub(
        r"RISK_PARITY_ASSET_CLASSES = \{.*?\n\}",
        make_block(classes),
        v20_src,
        count=1,
        flags=re.S,
    )


scenarios = [("基线 v0.20（6资产）", base_classes)]
for a in all_rp:
    c = {k: [x for x in v if x != a] for k, v in base_classes.items()}
    c = {k: v for k, v in c.items() if v}
    scenarios.append((f"剔除 {a}", c))
c_al = {k: (v + ["沪铝主连"] if k == "商品" else v) for k, v in base_classes.items()}
scenarios.append(("加回 沪铝（7资产）", c_al))
c_ta = {k: (v + ["PTA主连"] if k == "商品" else v) for k, v in base_classes.items()}
scenarios.append(("加回 PTA（7资产）", c_ta))
c_both = {k: (v + ["沪铝主连", "PTA主连"] if k == "商品" else v) for k, v in base_classes.items()}
scenarios.append(("加回 沪铝+PTA（=v0.19，8资产，自校验）", c_both))

rows = []
for label, classes in scenarios:
    if classes == base_classes:
        dus.STRATEGY_SCRIPT = V20_PATH
        tmp_path = None
    else:
        src2 = variant_src(classes)
        safe = re.sub(r"[^\w]", "", label)
        tmp_path = V20_PATH.parent / f"_jk_tmp_{safe}.py"
        tmp_path.write_text(src2, encoding="utf-8")
        dus.STRATEGY_SCRIPT = tmp_path
    try:
        bt = dus.build_backtest_result()
        gm = bt.df_metrics[
            (bt.df_metrics["回测区间"] == "全局 (Total)")
            & (bt.df_metrics["组合/资产"] == STRAT_NAME)
        ].iloc[0]
        rows.append(
            {
                "情景": label,
                "年化收益": gm["年化收益"],
                "年化波动": gm["年化波动"],
                "夏普比率": gm["夏普比率"],
                "最大回撤": gm["最大回撤"],
                "日度胜率": gm["日度胜率"],
            }
        )
        print(f"[done] {label}: 夏普={gm['夏普比率']} 回撤={gm['最大回撤']} 年化={gm['年化收益']}", flush=True)
    finally:
        if tmp_path is not None and tmp_path.exists():
            os.remove(tmp_path)

df = pd.DataFrame(rows)


def to_f(x):
    try:
        return float(str(x).strip("%")) / 100.0
    except Exception:
        return float(x)


base_sharpe = to_f(df.iloc[0]["夏普比率"])
print("\n===== 留一法敏感性结果（相对基线 v0.20）=====")
for _, r in df.iterrows():
    d = to_f(r["夏普比率"]) - base_sharpe
    arrow = "↑" if d > 0.001 else ("↓" if d < -0.001 else "=")
    print(
        f"{r['情景']:34s} 夏普={str(r['夏普比率']):>6s}({d:+.2f} {arrow})  "
        f"回撤={str(r['最大回撤']):>8s}  年化={str(r['年化收益']):>7s}  胜率={str(r['日度胜率']):>7s}"
    )
print("\n（说明：↑=比基线夏普更高，↓=更低；末行加回沪铝+PTA 应≈v0.19 的夏普 1.76，用于校验框架一致性）")
