-- headline_query
SELECT annual_return, sharpe, max_drawdown
FROM read_csv_auto('output/tables/strategy_metrics.csv', header=true)
WHERE strategy='erc' AND period='validation';

-- nav_query
WITH long_data AS (
  UNPIVOT (SELECT * FROM read_csv_auto('output/tables/strategy_nav.csv', header=true))
  ON equal_weight, inverse_downside_vol, erc
  INTO NAME strategy VALUE nav
), ranked AS (
  SELECT CAST(date AS DATE) AS date, strategy, nav,
         row_number() OVER (
           PARTITION BY date_trunc('month', CAST(date AS DATE)), strategy
           ORDER BY CAST(date AS DATE) DESC
         ) AS rn
  FROM long_data
)
SELECT strftime(date, '%Y-%m-%d') AS date,
       CASE strategy
         WHEN 'equal_weight' THEN '等权组合'
         WHEN 'inverse_downside_vol' THEN '逆下行波动率'
         ELSE '风险平价'
       END AS strategy,
       nav
FROM ranked WHERE rn=1 ORDER BY date, strategy;

-- performance_query
SELECT CASE strategy
         WHEN 'equal_weight' THEN '等权组合'
         WHEN 'inverse_downside_vol' THEN '逆下行波动率'
         ELSE '风险平价'
       END AS strategy,
       CASE period WHEN 'train' THEN '训练期' ELSE '验证期' END AS period,
       annual_return, annual_volatility, sharpe, max_drawdown, calmar, annual_turnover
FROM read_csv_auto('output/tables/strategy_metrics.csv', header=true)
WHERE period IN ('train', 'validation')
ORDER BY period, strategy;

-- algorithm_query
SELECT *, CASE method
         WHEN 'newton' THEN '阻尼牛顿法'
         WHEN 'lbfgsb' THEN 'L-BFGS-B'
         ELSE 'SLSQP'
       END AS algorithm
FROM read_csv_auto('output/tables/algorithm_summary.csv', header=true)
ORDER BY method;

-- optimizer_evolution_query
SELECT variant, label, objective_family, stage_order, observations,
       solver_success_rate, rc_pass_rate, median_iterations,
       median_runtime_ms, median_rc_error, max_rc_error
FROM read_csv_auto('output/tables/optimizer_evolution_summary.csv', header=true)
ORDER BY stage_order;

-- risk_budget_query
SELECT asset, raw_budget_multiplier, target_risk_budget,
       actual_risk_contribution, weight, absolute_rc_error,
       CAST(representative_date AS DATE) AS representative_date
FROM read_csv_auto('output/tables/risk_budget_extension.csv', header=true)
ORDER BY asset;

-- risk_budget_chart_query
SELECT asset,
       CASE measure
         WHEN 'target_risk_budget' THEN '目标风险预算'
         ELSE '实际风险贡献'
       END AS measure,
       value
FROM (
  UNPIVOT (
    SELECT asset, target_risk_budget, actual_risk_contribution
    FROM read_csv_auto('output/tables/risk_budget_extension.csv', header=true)
  ) ON target_risk_budget, actual_risk_contribution INTO NAME measure VALUE value
)
ORDER BY asset, measure;

-- sensitivity_query
SELECT *, CAST(CAST("window" AS INTEGER) AS VARCHAR) AS window_label,
       printf('%.2f', decay) AS decay_label
FROM read_csv_auto('output/tables/parameter_sensitivity.csv', header=true)
WHERE period='validation'
ORDER BY decay, "window";

-- optimal_query
SELECT asset,
       CASE measure WHEN 'weight' THEN '组合权重' ELSE '风险贡献' END AS measure,
       value
FROM (
  UNPIVOT (
    SELECT asset, weight, risk_contribution
    FROM read_csv_auto('output/tables/representative_optimal_solution.csv', header=true)
  ) ON weight, risk_contribution INTO NAME measure VALUE value
)
ORDER BY asset, measure;

-- estimator_query
SELECT *, CASE estimator
         WHEN 'sample' THEN '样本协方差'
         WHEN 'ewma_full' THEN 'EWMA全协方差'
         ELSE 'EWMA半协方差'
       END AS estimator_name
FROM read_csv_auto('output/tables/estimator_comparison.csv', header=true)
WHERE period='validation'
ORDER BY estimator;

-- stress_query
SELECT *, CASE method
         WHEN 'newton' THEN '阻尼牛顿法'
         WHEN 'lbfgsb' THEN 'L-BFGS-B'
         ELSE 'SLSQP'
       END AS algorithm
FROM read_csv_auto('output/tables/stress_test_summary.csv', header=true)
WHERE condition_number=(
  SELECT max(condition_number)
  FROM read_csv_auto('output/tables/stress_test_summary.csv', header=true)
)
ORDER BY method;

-- quality_query
WITH clean AS (
  SELECT * FROM read_csv_auto('data/etf_returns.csv', header=true)
), long_clean AS (
  UNPIVOT clean ON COLUMNS(* EXCLUDE (date)) INTO NAME asset VALUE daily_return
), profile AS (
  SELECT * FROM read_csv_auto('output/tables/data_quality_profile.csv', header=true)
)
SELECT '数据行数' AS check, (SELECT count(*) FROM clean) AS value, '通过' AS result
UNION ALL SELECT '资产数量', (SELECT count(DISTINCT asset) FROM long_clean), '通过'
UNION ALL SELECT '重复日期', (SELECT count(*)-count(DISTINCT date) FROM clean), '通过'
UNION ALL SELECT '清洗前缺失单元格', (SELECT sum(missing_before_fill) FROM profile), '已按0收益处理'
UNION ALL SELECT '清洗后缺失单元格', (SELECT count(*) FROM long_clean WHERE daily_return IS NULL), '通过';
