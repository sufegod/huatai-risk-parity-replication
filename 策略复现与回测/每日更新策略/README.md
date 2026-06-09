# 每日更新策略

本目录用于生成基于 v0.19 策略逻辑的每日回测报告。脚本运行时会先调用数据更新脚本，更新成功后再读取更新后的日涨跌幅数据，并加载 `策略复现与回测/策略代码/资产风险平价策略0.19（IC替换IM+日频胜率）.py` 中完整展示的回测代码，生成净值、指标、仓位明细、图表和 Markdown 报告。

## 运行方式

先在仓库根目录准备 Python 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如需更新最新数据，先准备本地敏感配置：

```powershell
Copy-Item .env.example .env
notepad .env
git check-ignore -v .env
```

`.env` 会由 `数据/日度收益数据更新/日度收益数据更新.py` 自动读取，不会提交到 GitHub。需要填写：

- `JYDB_DRIVER`：SQL Server ODBC 驱动名，默认 `ODBC Driver 17 for SQL Server`。
- `JYDB_SERVER`：JYDB SQL Server 地址。
- `JYDB_DATABASE`：数据库名，默认 `JYDB`。
- `JYDB_UID`：数据库用户名。
- `JYDB_PWD`：数据库密码。
默认运行。需要访问数据库时，使用沙箱外 PowerShell 入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\策略复现与回测\每日更新策略\run_daily_update.ps1
```

指定数据更新截止日：

```powershell
powershell -ExecutionPolicy Bypass -File .\策略复现与回测\每日更新策略\run_daily_update.ps1 -DataEndDate 2026-05-28
```

兼容旧命令行参数时：

```powershell
powershell -ExecutionPolicy Bypass -File .\策略复现与回测\每日更新策略\run_daily_update.ps1 -DataEndDate 2026-05-28 -ForceObservation
```

v0.19 日频调仓下，策略数据日期本身就是观察日，通常不需要使用 `--force-observation`。

仅基于现有 CSV 调试：

```powershell
python 策略复现与回测\每日更新策略\daily_update_strategy.py --skip-data-update
```

单独测试数据更新链路：

```powershell
powershell -ExecutionPolicy Bypass -File .\数据\日度收益数据更新\run_data_update.ps1 -EndDate 2026-05-28 -DryRun
```

不要把真实数据库密码写入 README、提交信息、issue、PR 或日志。`.env` 只用于线下分发和本地运行。

## 输入与输出

输入文件：

- `数据/日度收益数据更新/日涨跌幅_填充.csv`
- `数据/日度收益数据更新/日涨跌幅_未填充.csv`
- `数据/原始数据/股指期货信号.xlsx`

输出文件位于 `策略复现与回测/每日更新策略/输出`，并按用途分类：

- `仓位/仓位_YYYY-MM-DD.csv`：策略数据日期对应的新调仓目标；其中 `仓位来源观察日` 表示生成该仓位的日度观察日。
- `净值/策略每日净值走势_YYYY-MM-DD.csv`：完整回测每日净值。
- `指标/年度及全局回测指标_YYYY-MM-DD.csv`：全局与年度回测指标，包含每一行组合/资产的日度胜率和平均资金占用。
- `仓位明细/策略日度仓位明细_YYYY-MM-DD.csv`：日度观察日仓位明细，最后一列为资金占用比例。
- `图表/回测图表_YYYY-MM-DD.png`：累计净值、大类仓位、股指信号与仓位三联图。
- `报告/回测报告_YYYY-MM-DD.md`：包含核心指标、近期表现、当前仓位和输出文件清单的日报。

日期后缀使用实际参与策略计算的数据日期。报告中的 `日报观察日` 始终等于策略数据日期；`仓位来源观察日` 表示当前目标仓位来自哪一个日度观察日。脚本默认保持 v0.19 的日频调仓规则：每个实际交易日生成下一交易日目标仓位，完整历史回测也沿用该日频规则。报告与指标文件中的胜率字段为 `日度胜率`，口径为 `日收益 > 0` 的交易日数量除以全部统计交易日数量。
