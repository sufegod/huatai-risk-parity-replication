import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 解决图表中的中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 1. 读取 Excel 文件中指定的 "生产端通胀代理资产" sheet
df = pd.read_excel('代理资产行情.xlsx', sheet_name='生产端通胀代理资产', index_col=0, parse_dates=True)
df.index.name = 'Date'

# 选择所需的代理资产列
cols = ['布伦特原油', '南华螺纹钢指数', '动力煤平仓价']
# 注：若报错 KeyError，请检查 Excel 中这三列的表头名称是否包含多余空格
df = df[cols]

# 1.1 滚动 4 周移动平均 (约 20 个交易日) ，降低交易噪声干扰
df_ma = df.rolling(window=20, min_periods=1).mean()

# 1.2 取每周五的数值，得到周频收盘价序列
df_weekly = df_ma.resample('W-FRI').last()

# 2. 计算周频收盘价序列的周度环比收益率和年度同比收益率
wow_ret = df_weekly.pct_change(1)     # 环比（1周）
yoy_ret = df_weekly.pct_change(52)    # 同比（52周）

# （注意：生产端通胀指标均为多头序列，无需乘以 -1）

# 3. 计算滚动 3 年 (156周) 各代理资产年度同比收益率序列的标准差
std_3y = yoy_ret.rolling(window=156, min_periods=52).std()

# 3.1 对标准差的倒数归一化，作为权重序列
inv_std = 1 / std_3y
weights = inv_std.div(inv_std.sum(axis=1), axis=0)

# 4. 用权重序列对各代理资产周度环比收益率进行加权
# 使用上一期的权重（shift(1)）对当期收益率加权，避免未来函数
factor_wow = (weights.shift(1) * wow_ret).sum(axis=1, min_count=1)

# 剔除由于前期数据不足产生的空值
factor_wow_valid = factor_wow.dropna()

# 加 1 后累乘，还原为生产端通胀因子组合的全历史周频净值序列，并将起始点归一化为 1
factor_nav_all = (1 + factor_wow_valid).cumprod()
factor_nav_all = factor_nav_all / factor_nav_all.iloc[0]

# 计算因子的净值同比（52周）数据
factor_yoy_all = factor_nav_all / factor_nav_all.shift(52) - 1

# 5. 合并并提取从 2014 年开始的净值和净值同比数据
result = pd.DataFrame({
    'Inflation_Factor_NAV': factor_nav_all,
    'Inflation_Factor_YoY': factor_yoy_all
})

# 截取从 2014-01-01 开始的记录
result_2014 = result[result.index >= '2014-01-01'].copy()

# 将 2014 年的起始净值重新归为基准 1，方便图表展示
if not result_2014.empty:
    result_2014['Inflation_Factor_NAV'] = result_2014['Inflation_Factor_NAV'] / result_2014['Inflation_Factor_NAV'].iloc[0]

# --- 按照要求修改为中文命名 ---
# 将结果保存为 CSV 文件
csv_filename = '生产端通胀因子组合净值数据.csv'
result_2014.to_csv(csv_filename)
print(f"数据处理完毕，已保存至 {csv_filename}")

# 1. 生产端通胀因子组合净值走势图
plt.figure(figsize=(12, 6))
plt.plot(result_2014.index, result_2014['Inflation_Factor_NAV'], label='生产端通胀因子组合净值', color='purple')
plt.title('生产端通胀因子组合净值走势 (2014至今)')
plt.xlabel('日期')
plt.ylabel('净值 (2014年基准=1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('生产端通胀因子组合净值走势图.png')
plt.close()

# 2. 生产端通胀因子组合净值同比走势图
plt.figure(figsize=(12, 6))
plt.plot(result_2014.index, result_2014['Inflation_Factor_YoY'], label='生产端通胀因子组合净值同比', color='orange')
plt.axhline(0, color='black', linestyle='--', linewidth=1) # 绘制 0 轴辅助线
plt.title('生产端通胀因子组合净值同比走势 (2014至今)')
plt.xlabel('日期')
plt.ylabel('同比收益率')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('生产端通胀因子组合净值同比走势图.png')
plt.close()