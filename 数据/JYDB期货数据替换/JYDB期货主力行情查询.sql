-- JYDB 期货主力行情查询 SQL 模板
-- 数据库: JYDB
-- 日期范围: 2012-11-05 至 2026-03-31; 输出按原 Excel 日期 2013-01-04 至 2026-03-31 对齐
-- 注意: 实际抽取使用参数化查询；此文件不包含数据库密码。

-- 金融期货: dbo.Fut_TradingQuote
SELECT
    TradingDay AS 日期,
    ContractInnerCode AS 合约内部编码,
    ContractCode AS 合约代码,
    CAST(ClosePrice AS float) AS 收盘价,
    CAST(SettlePrice AS float) AS 结算价,
    MainContractMark AS 主力标志
FROM dbo.Fut_TradingQuote
WHERE ExchangeCode = ?
  AND OptionCode = ?
  AND TradingDay BETWEEN ? AND ?
  AND ClosePrice IS NOT NULL
ORDER BY TradingDay, ContractInnerCode;

-- 商品/黄金/原油期货: dbo.Fut_DailyQuote + dbo.Fut_ContractMain
SELECT
    q.EndDate AS 日期,
    q.InnerCode AS 合约内部编码,
    COALESCE(cm.ContractCode, q.ContractName, CONVERT(varchar(50), q.InnerCode)) AS 合约代码,
    CAST(q.ClosePrice AS float) AS 收盘价,
    CAST(q.SettlePrice AS float) AS 结算价,
    q.MainContractMark AS 主力标志
FROM dbo.Fut_DailyQuote AS q
LEFT JOIN dbo.Fut_ContractMain AS cm
  ON cm.ContractInnerCode = q.InnerCode
WHERE q.Exchange = ?
  AND q.OptionCode = ?
  AND q.EndDate BETWEEN ? AND ?
  AND q.ClosePrice IS NOT NULL
ORDER BY q.EndDate, q.InnerCode;

-- 品种映射
-- 资产名称,来源表,交易所代码,品种代码,品种前缀,说明
沪深300主连,dbo.Fut_TradingQuote,20,3145,IF,沪深300股指期货
10年国债主连,dbo.Fut_TradingQuote,20,502,T,10年期国债期货
沪金主连,dbo.Fut_DailyQuote,10,313,AU,上海黄金期货
豆粕主连,dbo.Fut_DailyQuote,13,345,M,大商所豆粕期货
中证1000主连,dbo.Fut_TradingQuote,20,39144,IM,中证1000股指期货
30年国债主连,dbo.Fut_TradingQuote,20,504,TL,30年期国债期货
沪铜主连,dbo.Fut_DailyQuote,10,305,CU,上海铜期货
沪铝主连,dbo.Fut_DailyQuote,10,310,AL,上海铝期货
PTA主连,dbo.Fut_DailyQuote,15,322,TA,郑商所PTA期货
原油主连,dbo.Fut_DailyQuote,11,319,SC,上海原油期货
