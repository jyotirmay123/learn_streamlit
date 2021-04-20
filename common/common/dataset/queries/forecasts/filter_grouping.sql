SELECT
  optimization_group,
  optimization_size,
  optimization_value,
  ANY_VALUE(scope_bits) AS scope_bits,
  SUM(predicted_revenue) AS predicted_revenue,
  SUM(predicted_profit) AS predicted_profit,
FROM
  `{{ filtered_table }}`
GROUP BY
  optimization_group, optimization_size, optimization_value
