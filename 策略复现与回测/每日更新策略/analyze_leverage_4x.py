# -*- coding: utf-8 -*-
"""4 倍杠杆情景压力测试

【研究性质，不接生产链路】
问题：如果把这个策略用 4 倍杠杆来做，过程中会遇到什么问题？

方法：
  1. 用 2016 起回溯的真实日收益（上证50ETF替代口径）派生 4x 杠杆日收益
  2. 逐日跟踪净值，找最大回撤、最长水下期
  3. 按每日实际保证金占用计算"维持保证金率"，模拟强平触发
  4. 检查单日极端损失、连续亏损、流动性/成本放大
  5. 逐年分解：哪些年份会出事

口径说明（务必写清，避免误解）：
  - 简化为"总敞口 = 4 × 净资产"，即组合内所有资产同比例放大到 4 倍
  - 现实中更接近的做法是：期货保证金占用从 ~16.6% 拉到 4 倍（~66.4%），
    剩余 33.6% 现金；但 ETF 部分（红利低波/上证50）本身无杠杆，
    想在 ETF 上做 4 倍必须用融资或期权，成本更高
  - 本脚本按"整体 4 倍"算，是最直接的解读；另附"仅期货端 4 倍"作对照
"""
import pandas as pd, numpy as np

P = 'C:/Users/aa/WorkBuddy/2026-05-28-14-41-45/huatai-risk-parity-replication'
D = f'{P}/策略复现与回测/每日更新策略'

# ---------- 读入真实日收益（'上证50ETF_日收益' 列实为策略日收益）----------
d = pd.read_csv(f'{D}/输出/净值/2016起回溯_日收益对比.csv', index_col=0, parse_dates=True)
ret = d['上证50ETF_日收益'].dropna()

# 日度保证金占用（券商口径）
pos = pd.read_csv(f'{D}/输出/对比分析/2016起_仓位明细_上证50.csv', index_col=0, parse_dates=True)
margin_daily = pos['margin'].reindex(ret.index).ffill()

print('=' * 100)
print('4 倍杠杆情景压力测试（研究性质，基于 2016 起回溯真实日收益）')
print('=' * 100)
print(f'样本区间：{ret.index[0].date()} ~ {ret.index[-1].date()}（{len(ret)} 个交易日 / {len(ret)/243:.1f} 年）')
print(f'无杠杆基准：累计 {(1+ret).prod()-1:+.2%}，年化 {((1+ret).prod()**(243/len(ret))-1):+.2%}，'
      f'年化波动 {ret.std()*np.sqrt(243):.2%}')

# ================= 1. 杠杆日收益构造 =================
# 关键假设：每日再平衡回目标杠杆（否则杠杆会漂移）
# 期货端杠杆不额外占用现金；现金收益按 (1 - 4×保证金占用) 计，可为负（融资成本）
L = 4.0

# -- 口径A：整体 4 倍（最直观）--
levA = L * ret

# -- 口径B：期货端 4 倍，ETF 部分不放大 --
# 拆出 ETF 权重与其余（期货）权重，各自处理
ETF_COLS = ['上证50ETF']
fut_w = pos[[c for c in pos.columns if c not in ETF_COLS + ['signal', 'idxw', 'margin']]].sum(axis=1)
etf_w = pos[[c for c in ETF_COLS if c in pos.columns]].sum(axis=1)
fut_w = fut_w.reindex(ret.index).ffill()
etf_w = etf_w.reindex(ret.index).ffill()
# 期货端名义敞口 ×4，ETF 保持 1 倍（但 ETF 那部分钱也要占用现金）
# 近似：杠杆收益 ≈ (4×期货权重 + 1×ETF权重) 的资产收益 − 融资成本
# 此处简化用整体比例：可放大比例 ≈ (1 - etf_w)，杠杆作用在期货上
scaleB = 1 + (L - 1) * (1 - etf_w) / (1 - etf_w + etf_w + 1e-12)
# 更保守且清晰的近似：敞口倍数 = 1 + 3×(1−etf_w)
exposureB = 1 + (L - 1) * (1 - etf_w)
levB = exposureB * ret

# ================= 2. 全期表现对比 =================
def perf(r, label):
    nav = (1 + r).cumprod()
    yrs = len(r) / 243
    tot = nav.iloc[-1] - 1
    ann = nav.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(243)
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    shp = ann / vol if vol > 0 else np.nan
    return dict(口径=label, 累计=tot, 年化=ann, 波动=vol, 夏普=shp,
                最大回撤=mdd, 卡玛=(ann / abs(mdd)) if mdd != 0 else np.nan,
                最差单日=r.min(), 最好单日=r.max())

rows = [perf(ret, '① 无杠杆（基准）'),
        perf(levA, f'② 整体{L:.0f}x'),
        perf(levB, f'③ 仅期货端{L:.0f}x')]
res = pd.DataFrame(rows)
print('\n' + '=' * 100)
print('一、全期表现对比')
print('=' * 100)
show = res.copy()
for c in ['累计', '年化', '波动', '最大回撤', '最差单日', '最好单日']:
    show[c] = show[c].map(lambda x: f'{x:+.2%}')
for c in ['夏普', '卡玛']:
    show[c] = show[c].map(lambda x: f'{x:.2f}')
print(show.to_string(index=False))

# ================= 3. 4x 下的回撤路径与"爆仓" =================
r4 = levA
nav4 = (1 + r4).cumprod()
dd4 = nav4 / nav4.cummax() - 1
mdd4 = dd4.min()
mdd4_date = dd4.idxmin()
nav_base = (1 + ret).cumprod()
dd_base = nav_base / nav_base.cummax() - 1

print('\n' + '=' * 100)
print(f'二、{L:.0f}x 杠杆下的回撤（关键：多久回本 / 会不会归零）')
print('=' * 100)
print(f'  最大回撤        {mdd4:+.2%}   （无杠杆 {dd_base.min():+.2%}，放大 {mdd4/dd_base.min():.2f} 倍）')
print(f'  最深日期        {mdd4_date.date()}')
print(f'  最低净值        {nav4.min():.4f}')

# 水下期（从峰值回到峰值）
underwater = dd4 < -1e-9
groups = (underwater != underwater.shift()).cumsum()
uw = underwater.groupby(groups).sum()
uw = uw[underwater.groupby(groups).first()]
if len(uw) > 0:
    uw_days = uw.sort_values(ascending=False).head(5)
    print(f'\n  最长的 5 段"水下期"（交易日数）：')
    for gid, n in uw_days.items():
        seg = underwater[groups == gid]
        print(f'    {seg.index[0].date()} ~ {seg.index[-1].date()}  {int(n):>4} 个交易日（约 {n/243:.2f} 年）')

# 回撤超过关键阈值的次数
for thr in [0.20, 0.30, 0.40, 0.50, 0.70, 0.75]:
    n = (dd4 <= -thr).sum()
    print(f'  回撤曾触及 -{thr:.0%}：{"是" if n > 0 else "否":<3} （{n} 个交易日，占 {n/len(dd4):.1%}）')

# ================= 4. 保证金与强平模拟 =================
print('\n' + '=' * 100)
print('三、保证金与强平风险（这是 4 倍杠杆最可能出问题的地方）')
print('=' * 100)
# 无杠杆时保证金占用均值 ~16.6%；4 倍即 ~66.4%
mg_1x = margin_daily.mean()
mg_4x = mg_1x * L
print(f'  无杠杆平均保证金占用   {mg_1x:.2%}')
print(f'  {L:.0f}x 下平均保证金占用 {mg_4x:.2%}  → 剩余现金 {1-mg_4x:.2%}')
print(f'  单日现货占比（最坏）    {margin_daily.max():.2%} × {L:.0f} = {margin_daily.max()*L:.2%}')
print()
# 强平逻辑：净值跌到"维持保证金"以下会被追保/强平
# 假设维持保证金率 = 保证金占用 × 0.75（交易所通常要求维持保证金低于开仓保证金）
MAINT_RATIO = 0.75
# 可用缓冲 = 1 - 保证金占用（现金部分）→ 净值需跌多少才触发追保
# 简化：维持保证金 = mg_4x × MAINT_RATIO；现金 = 1 - mg_4x
# 触发追保当：净值 × (1 - mg_4x × MAINT_RATIO) < ... 用共同市场语言：
# 剩余现金 / 保证金占用 = 缓冲比例。跌掉该比例即触及维持线。
buffer_1x = (1 - mg_1x) / mg_1x
buffer_4x = (1 - mg_4x) / mg_4x
print(f'  缓冲倍数（现金 ÷ 保证金占用）：无杠杆 {buffer_1x:.2f}x → 可承受 {buffer_1x:.1%} 资产跌幅')
print(f'                             {L:.0f}x  {buffer_4x:.2f}x → 可承受仅 {buffer_4x:.1%} 资产跌幅')
print()
# 逐日滚动：放大 4 倍的期货端亏损会吃掉现金吗
# 期货端每日盈亏 = 4 × (期货权重部分收益)，现金端 = 1 - 4×保证金
fut_ret_daily = (pos[[c for c in pos.columns if c not in ETF_COLS + ['signal','idxw','margin']]]
                 .mul(ret, axis=0).sum(axis=1)).reindex(ret.index).fillna(0)
fut_only = pos[[c for c in pos.columns if c not in ETF_COLS + ['signal','idxw','margin']]]
wsum = fut_only.sum(axis=1).reindex(ret.index).ffill()
fut_contrib_ret = (ret * (wsum / wsum)) # placeholder
# 直接用：期货端名义权重 wsum → 4 倍杠杆后敞口 4*wsum
fut_ret_4x = ret * (L * wsum) / (L * wsum + etf_w + (1 - L * mg_1x) + 1e-12)

# 更直接的做法：计算"累计最大连续亏损"来评估追保压力
def max_consec_loss(r):
    neg = r < 0
    g = (neg != neg.shift()).cumsum()
    losses = r[neg].groupby(g[neg]).sum()
    return losses.min(), int(neg.groupby(g).sum().max())

cl4, n4 = max_consec_loss(r4)
cl1, n1 = max_consec_loss(ret)
print(f'  最长连续亏损：无杠杆 {n1} 日 / 累计 {cl1:+.2%}')
print(f'              {L:.0f}x  {n4} 日 / 累计 {cl4:+.2%}')

# ================= 5. 逐年分解 =================
print('\n' + '=' * 100)
print(f'四、逐年分解：{L:.0f}x 杠杆下哪些年份"会出事"')
print('=' * 100)
yr_rows = []
for y in sorted(set(ret.index.year)):
    m = ret.index.year == y
    if m.sum() < 20:
        continue
    r1 = ret[m]
    rl = r4[m]
    nav1 = (1 + r1).cumprod()
    navl = (1 + rl).cumprod()
    dd1 = (nav1 / nav1.cummax() - 1).min()
    ddl = (navl / navl.cummax() - 1).min()
    yr_rows.append({
        '年份': y,
        '无杠杆收益': (1 + r1).prod() - 1,
        f'{L:.0f}x收益': (1 + rl).prod() - 1,
        '无杠杆最大回撤': dd1,
        f'{L:.0f}x最大回撤': ddl,
        f'{L:.0f}x最差单日': rl.min(),
        '爆仓?': '⚠️ 是' if ddl <= -1.0 else ('危险' if ddl <= -0.5 else ''),
    })
yr = pd.DataFrame(yr_rows)
yshow = yr.copy()
for c in yshow.columns:
    if c not in ['年份', '爆仓?']:
        yshow[c] = yshow[c].map(lambda x: f'{x:+.2%}')
print(yshow.to_string(index=False))

# ================= 6. 极端单日 =================
print('\n' + '=' * 100)
print(f'五、极端单日（{L:.0f}x 下）')
print('=' * 100)
worst = r4.nsmallest(10)
print(f"{'日期':<12}{'无杠杆':>10}{'4x':>10}")
for dt, v in worst.items():
    print(f"{str(dt.date()):<12}{ret[dt]:>10.2%}{v:>10.2%}")
print(f'\n  4x 下单日跌幅 >5% 的有 {(r4 <= -0.05).sum()} 天；>8% 的有 {(r4 <= -0.08).sum()} 天；>10% 的有 {(r4 <= -0.10).sum()} 天')

# ================= 7. 杠杆下的"数学陷阱" =================
print('\n' + '=' * 100)
print('六、杠杆的数学陷阱（为什么"4倍收益"拿不到）')
print('=' * 100)
ann1 = ((1 + ret).prod() ** (243 / len(ret)) - 1)
vol1 = ret.std() * np.sqrt(243)
print(f'  无杠杆：年化收益 {ann1:.2%}，年化波动 {vol1:.2%}')
print(f'  理论 4x：年化收益 {4*ann1:.2%}，年化波动 {4*vol1:.2%}')
print(f'  实际 4x（按日复利）：年化 {((1+r4).prod()**(243/len(r4))-1):.2%}')
print(f'  → 差异主因：波动拖累（volatility drag）≈ -½ × σ² × (L²-L)')
drag = -0.5 * vol1**2 * (L**2 - L)
print(f'     估算波动拖累 {drag:.2%}/年')
print(f'  融资成本：4x 需额外借 3 倍，按年化 3% 计 ≈ {3*0.03:.2%}/年（未计入上方）')

# ================= 8. 保存 =================
out = f'{D}/输出/对比分析/杠杆情景_4x压力测试_2016-2026.csv'
yr.to_csv(out, index=False, encoding='utf-8-sig')
print(f'\n已导出：{out}')

# 保存净值序列供画图
navdf = pd.DataFrame({'无杠杆': nav_base, '4x': nav4,
                      '无杠杆回撤': dd_base, '4x回撤': dd4})
navdf.to_csv(f'{D}/输出/净值/杠杆情景_4x净值序列.csv', encoding='utf-8-sig')
print(f'已导出：{D}/输出/净值/杠杆情景_4x净值序列.csv')
