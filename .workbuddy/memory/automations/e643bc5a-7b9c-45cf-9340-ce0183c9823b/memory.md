# 自动化执行记录 — WorkBuddy 增量备份到 E 盘

任务：把 `C:\Users\aa\WorkBuddy` 与 `C:\Users\aa\.workbuddy` 增量备份到 `E:\WorkBuddy_Backup\`（滚动，只增不删，禁 /MIR）。
方式：PowerShell 内联 robocopy（/E /XJ /R:1 /W:1 /MT:16），日志写 C 盘 Temp；末尾 exit 0。

## 环境要点（复用）
- 历史 memory 曾在首次执行时缺失 → 直接开跑即可，不要中断询问。
- E 盘目标根已存在遗留测试项：`_t5dst\`、`_logA.txt`、`_logC.txt`、`_log_*.txt`、`_t5log.txt`、`_write_probe.txt`。沙箱禁删 E 盘文件，只在汇报里提一句。
- `E:\WorkBuddy_Backup\dot_workbuddy\workspace\` 是**早期未加排除清单时**留下的 10 万+ 文件遗留副本；源端现已排除 workspace，故它会长期作为 dest extras 存在（导致 rc2 常带 +2 位）。不得删除。
- 统计口径：`Get-ChildItem -Recurse -File -Force | Measure-Object Length -Sum`；文件数取 `@(...).Count`，两边目录分别统计后求和。
- 耗时极短（增量，NVMe 级 E 盘）：本次两次 robocopy 合计约 1–2 秒，Stopwatch 读数 0.3s。

## 执行历史
### 2026-09-29 15:30（首次记录）
- E 盘可用。
- rc1 = 1（WorkBuddy：有文件复制）→ 复制 30 文件 / 5.48 MB；源 6746 文件 / 399.33 MB。
- rc2 = 3（dot_workbuddy：有复制 + 目标端有多余文件）→ 复制 68 文件 / 36.15 MB；源（排除后）1845 文件 / 198.50 MB；其他(extras)=10。
- 本次新增/更新 = 30 + 68 = **98** 文件；备份总量 **822.8 MB / 107828 文件**（WorkBuddy 399.3 MB + dot_workbuddy 423.4 MB）；E 盘剩余 **295.2 GB**。
- 已建 `E:\WorkBuddy_Backup\_备份日志.txt`（本次为首次创建，含表头）并追加本次一行。
- 无失败项（失败列 = 0）。耗时远小于 1 分钟，排除清单有效。
