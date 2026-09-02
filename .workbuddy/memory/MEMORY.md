# 项目长期记忆 — huatai-risk-parity-replication

## 仓库 Git 环境（2026-09-02 确认）
- **坑：本地远程追踪引用 `refs/remotes/origin/main` 已失效**，停在旧值 `8a54e75`（2026-05-28），且 git 无法写入更新（`git fetch` 打印成功、`git update-ref` 返回 0，但引用不变；`.git/refs/remotes/origin/` 目录会被清掉）。
- **后果**：`git status -sb` 会虚报 `ahead 29/30`，这是**幽灵领先**，不代表真的有那么多提交没推。
- **正确做法**：用 `git ls-remote origin` 取远程真实 SHA，再 `git log --oneline <远程SHA>..main` 看真实待推送提交。不要相信 `origin/main`。
- **GitHub 链路不稳定**：push 常报 `CONNECT tunnel failed, response 502` 或 `schannel: server closed abruptly (missing close_notify)`；严重时连 `ls-remote` 也 502（透明代理故障，git config 与 env 均未配代理）。属临时性故障，换个时间重试往往能成功（8/31 曾成功推送）。
- GitHub SSH(443) 网络可达，但本机**未配置 SSH key**（`Permission denied (publickey)`），暂不能作为 https 的备份通道。

## 策略与流程约定
- 「更新策略」= 运行 `策略复现与回测/每日更新策略/daily_update_strategy.py`（venv: `C:/Users/aa/.workbuddy/binaries/python/envs/default/Scripts/python.exe`），更新数据至最新交易日 + 重算回测。
- 每日 15:30 自动化任务（automation-1787648434423）：更新 → 读报告 → add/commit → 尝试 push；push 失败仅汇报不阻塞。
- 当前策略版本 v0.19（IC 替换 IM + 剔除 30 年国债 TL，10 年国债权重约 53%）。
- 报告输出目录：`策略复现与回测/每日更新策略/输出/报告/回测报告_YYYY-MM-DD.md`（另有 仓位/净值/指标/仓位明细/图表 子目录）。
- 报告默认只列 top 8 持仓；需要完整 10 大持仓时读同日期的 `输出/仓位/仓位_YYYY-MM-DD.csv`。
