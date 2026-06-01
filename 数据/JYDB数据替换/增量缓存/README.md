# 增量缓存目录

本目录用于保存 `update_daily_returns.py` 从 JYDB 和 iFinD 拉取后的累计原始输入数据。

脚本首次正常更新时会生成：

- `期货行情.csv`
- `红利低波ETF行情.csv`
- `GC001.csv`

这些缓存文件由脚本维护，重叠日期会以新拉取数据覆盖旧缓存。不要手工修改缓存文件；如需重新初始化，请运行：

```powershell
python 数据\JYDB数据替换\update_daily_returns.py --full-refresh
```
