# 每日更新策略

本目录用于生成基于 v0.16 策略逻辑的每日回测报告。脚本运行时会先调用数据更新脚本，更新成功后再读取更新后的日涨跌幅数据，完整生成净值、指标、仓位明细、图表和 Markdown 报告。

## 运行方式

默认运行：

```powershell
python 策略复现与回测\每日更新策略\daily_update_strategy.py
```

指定数据更新截止日：

```powershell
python 策略复现与回测\每日更新策略\daily_update_strategy.py --data-end-date 2026-05-28
```

短周或节假日前需要把当前数据日视为周度观察日时：

```powershell
python 策略复现与回测\每日更新策略\daily_update_strategy.py --data-end-date 2026-05-28 --force-observation
```

仅基于现有 CSV 调试：

```powershell
python 策略复现与回测\每日更新策略\daily_update_strategy.py --skip-data-update
```

## 输入与输出

输入文件：

- `数据/JYDB数据替换/日涨跌幅_填充.csv`
- `数据/JYDB数据替换/日涨跌幅_未填充.csv`
- `数据/原始数据/股指期货信号.xlsx`

输出文件位于 `策略复现与回测/每日更新策略/输出`，并按用途分类：

- `仓位/仓位_YYYY-MM-DD.csv`：策略数据日期对应的当前仓位或新调仓目标。
- `净值/策略每日净值走势_YYYY-MM-DD.csv`：完整回测每日净值。
- `指标/年度及全局回测指标_YYYY-MM-DD.csv`：全局与年度回测指标。
- `仓位明细/策略周度仓位明细_YYYY-MM-DD.csv`：周度观察日仓位明细。
- `图表/回测图表_YYYY-MM-DD.png`：累计净值、大类仓位、股指信号与仓位三联图。
- `报告/回测报告_YYYY-MM-DD.md`：包含核心指标、近期表现、当前仓位和输出文件清单的日报。

日期后缀使用实际参与策略计算的数据日期。脚本默认保持 v0.16 的周频调仓规则：周五生成新的下一交易日目标仓位，非周五报告当前有效仓位。完整历史回测始终沿用 v0.16 原始周频规则，不受 `--force-observation` 影响。
