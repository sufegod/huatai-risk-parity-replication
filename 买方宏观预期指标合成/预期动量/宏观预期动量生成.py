import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 解决图表中的中文显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False


# 1. 定义位移路径比动量的计算函数
def calc_displacement_path_momentum(series, n):
    """
    计算位移路径比动量 (Displacement-Path Ratio Momentum)
    """
    displacement = series.diff(n)
    abs_diff = series.diff(1).abs()
    path = abs_diff.rolling(window=n).sum()

    # 避免分母为 0 的情况，用 np.where 处理
    momentum = np.where(path == 0, 0, displacement / path)
    return pd.Series(momentum, index=series.index)


# 2. 读取之前生成的因子净值数据
try:
    df_growth = pd.read_csv('../增长高频因子/增长因子组合净值数据.csv', index_col=0, parse_dates=True)
    df_inflation = pd.read_csv('../生产端通胀高频因子/生产端通胀因子组合净值数据.csv', index_col=0, parse_dates=True)
except FileNotFoundError as e:
    print("错误: 未找到因子净值文件，请确保您已执行前两段代码并在同目录下生成了相关的 CSV 文件。")
    raise e

# 3. 设定参数 N (过去一个月对应 4 周)
N_weeks = 4

# 4. 计算预期动量 (基于同比 YoY 数据)
growth_momentum = calc_displacement_path_momentum(df_growth['Growth_Factor_YoY'], n=N_weeks)
inflation_momentum = calc_displacement_path_momentum(df_inflation['Inflation_Factor_YoY'], n=N_weeks)

# 将计算结果整理为 DataFrame，并直接使用要求的中文列名
momentum_df = pd.DataFrame({
    '增长预期动量': growth_momentum,
    '通胀预期动量': inflation_momentum
}).dropna()  # 剔除由于计算滚动窗口产生的初始空值

# 5. 保存数据到 CSV
csv_filename = '增长预期动量与通胀预期动量数据.csv'
momentum_df.to_csv(csv_filename)
print(f"数据处理完毕，动量指标已保存至 {csv_filename}")

# 6. 绘制并保存动量走势图

# --- 绘制：增长预期动量走势图 ---
plt.figure(figsize=(12, 6))
plt.plot(momentum_df.index, momentum_df['增长预期动量'], label='增长预期动量', color='blue', linewidth=1.5)
plt.axhline(0, color='black', linestyle='-', linewidth=1)  # 零轴
plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)  # 极值上限
plt.axhline(-1, color='gray', linestyle='--', linewidth=0.8)  # 极值下限
plt.title('增长预期动量走势图 (1个月窗口)')
plt.xlabel('日期')
plt.ylabel('动量值 [-1, 1]')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('增长预期动量走势图.png')
plt.close()
print("图表已保存: 增长预期动量走势图.png")

# --- 绘制：通胀预期动量走势图 ---
plt.figure(figsize=(12, 6))
plt.plot(momentum_df.index, momentum_df['通胀预期动量'], label='通胀预期动量', color='purple', linewidth=1.5)
plt.axhline(0, color='black', linestyle='-', linewidth=1)  # 零轴
plt.axhline(1, color='gray', linestyle='--', linewidth=0.8)  # 极值上限
plt.axhline(-1, color='gray', linestyle='--', linewidth=0.8)  # 极值下限
plt.title('通胀预期动量走势图 (1个月窗口)')
plt.xlabel('日期')
plt.ylabel('动量值 [-1, 1]')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('通胀预期动量走势图.png')
plt.close()
print("图表已保存: 通胀预期动量走势图.png")