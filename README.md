# 华泰风险平价策略复现

本项目用于整理和复现华泰风险平价相关策略，包含宏观预期指标合成、风险平价策略回测、回测结果图表和参考研报。

## 环境要求

- Python 3.11 或更高版本。每日数据更新脚本使用标准库 `tomllib`，低于 3.11 的 Python 不能直接运行。
- Python 依赖见 `requirements.txt`。
- 每日更新最新数据需要能访问 JYDB SQL Server、iFinD MCP，并安装 Microsoft ODBC Driver for SQL Server。当前默认驱动名为 `ODBC Driver 17 for SQL Server`。
- 绘图脚本使用中文字体；如图表中文乱码，请安装 `SimHei`、`Microsoft YaHei`、`WenQuanYi Micro Hei` 或 `Noto Sans CJK`。

## 快速启动

从仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests
```

仅使用仓库内已有 CSV 复现每日策略报告，不更新外部数据：

```powershell
python 策略复现与回测\每日更新策略\daily_update_strategy.py --skip-data-update
```

生成最新数据并运行每日策略：

```powershell
python 策略复现与回测\每日更新策略\daily_update_strategy.py --data-end-date 2026-05-28
```

不指定 `--data-end-date` 时，数据更新脚本会尝试使用各数据源共同可得的最新日期。

## 目录结构

- `买方宏观预期指标合成/`: 高频宏观因子、预期动量因子相关脚本、数据和图表。
- `策略复现与回测/`: 风险平价策略回测脚本、输入数据、回测图表和回测指标。
- `策略研报来源/`: 策略复现使用的参考研报。

## 本地敏感配置

仓库根目录支持本地 `.env` 文件，脚本启动时会自动读取。`.env` 已在 `.gitignore` 中，不会提交到 GitHub；`.env.example` 可以提交，用于说明字段。

创建本地配置：

```powershell
Copy-Item .env.example .env
notepad .env
git check-ignore -v .env
```

`.env` 字段：

- `JYDB_DRIVER`: SQL Server ODBC 驱动名，默认 `ODBC Driver 17 for SQL Server`。
- `JYDB_SERVER`: JYDB SQL Server 地址。
- `JYDB_DATABASE`: 数据库名，默认 `JYDB`。
- `JYDB_UID`: 数据库用户名。
- `JYDB_PWD`: 数据库密码。
- `IFIND_MCP_URL`: iFinD MCP 地址。
- `IFIND_MCP_AUTHORIZATION`: iFinD MCP 的完整 `Authorization` 值。

安全要求：

- 不要把真实密码、token、Authorization 写入 README、issue、PR、提交信息或日志。
- 不要提交 `.env`；线下分发给需要使用的人。
- 如果 `.env` 缺少 iFinD 配置，脚本会回退读取 `--ifind-config` 指定的 TOML，默认是 `~/.codex/config.toml`。

## 发布说明

项目包含 Excel 数据、PDF 研报和回测图片。若计划公开发布到 GitHub，请先确认数据和研报的授权；如仅自用，建议创建 private 仓库。
