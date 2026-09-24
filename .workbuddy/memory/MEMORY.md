# 项目长期记忆 — huatai-risk-parity-replication

## Git 环境与同步
- 本地追踪引用 `refs/remotes/origin/main` 已失效且无法更新，`git status -sb` 会虚报 "ahead N"（幽灵领先）。正确做法：`git ls-remote origin` 取真实 SHA → `git log <SHA>..main` 看真实积压。
- GitHub HTTPS 链路不稳定（偶发 502 / schannel close_notify / Connection reset），呈同日间歇性；**失败不要改 git 配置**，隔几十分钟原样重试即可。SSH(443) 可达但未配 key，暂不能作备份通道。
- **Shell PATH 损坏绕行**（2026-09-14 起）：git/coreutils 命令一律走 PortableGit login shell：`/c/Users/aa/.workbuddy/binaries/PortableGit/versions/1.2.0/bin/bash.exe -lc "cd <repo> && git ..."`。push 等慢操作必须 `run_in_background=true`（19MB csv 前台会 SIGTERM）。Python 可直接用绝对路径。

## 策略与每日流程
- 当前生产版 **v0.19**（IC 替换 IM + 剔除 30 年国债 TL，10 年国债权重约 60%）。
- 更新 = 运行 `策略复现与回测/每日更新策略/daily_update_strategy.py`（venv：`C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe`），更新数据至最新交易日 + 重算回测。
- **09:00 自动化**（automation-1787648434423）：更新→读报告→add/commit（信息含版本/数据日期/净值/夏普）→尝试 push；push 失败仅汇报不阻塞。git 身份已配（aa/aa@workbuddy.local），勿 pull。
- 输出目录：`输出/{报告,净值,仓位,指标,仓位明细,图表}/`，报告默认列 top 8 持仓，完整 10 大读 `输出/仓位/仓位_YYYY-MM-DD.csv`。

## 存储格式
- 每日产物全为 **CSV**（无 Excel 生成）。数据层覆盖式（全量集中）：`数据/日度收益数据更新/` 下 3 个 csv + `增量缓存/` 下 4 个 csv（含 19MB 期货行情.csv）。
- 策略产物按日期新建不覆盖：`输出/净值|仓位|仓位明细|指标|报告|图表/..._YYYY-MM-DD.*`。
- `.gitignore` 忽略 `输出/` ⇒ 报告/图表不入库、不随 push 同步，须本机查看。增量缓存 csv 已被跟踪，修改仍进提交。

## 股指信号（人工维护输入）
- `数据/原始数据/股指期货信号.xlsx` 由 `load_index_signal()` 读取并 `.ffill()` 前向填充；**只在变更日填值**，其余留空正常（留空≠漏填）。
- 映射：`INDEX_BASE_WEIGHT=0.30`；仓位=0.30×normalize(signal)，signal≤0→清仓，∈(0,1]→0~30%。算例 0.7→21%(IF/IC 各半)、0.5→15%、0.3→9%。
- 变更轨迹：08-27=0.7 → 09-16=0.5 → 09-17=0.3(ffill 至 09-21) → **09-22=0.5**（当前生效，→15% 仓位）。
- 信号为主观打分，无法自动推导；距上次更新 >15 天可提醒补录，但日常汇报不反复提示"信号陈旧"。

## 数据源与连接
- 核心链路：JYDB(192.168.10.48:1433) → `日度收益数据更新.py` → 本地 CSV → 策略（策略不连库）。`JYDB`(主库)/`FTDB`(GC001)。pyodbc，账号 tsreadonly 只读，密码从仓库根 `.env` 读（含 JYDB_PWD/SERVER/DATABASE/UID/DRIVER）。
- **ODBC 驱动坑**：本机已无 "ODBC Driver 17"，`.env` 的 `JYDB_DRIVER` 已改 `"SQL Server"`（旧驱动可连且兼容 Encrypt=no）。若重装 17/18 可改回。
- 独立数据源：新浪财经（akshare），仅用于商品趋势模型，与原策略无关。

## 隔离与清理
- 商品趋势模型数据隔离副本在 `商品趋势模型/历史数据/`（单向源→副本，禁反向回写）。
- 2026-09-10 已删「趋势择时 v0.20 系列」20 文件（备份 `_deleted_trend_timing_backup_2026-09-10`）。现存谱系 v0.16/17/18/19，v0.19 生产版。
