WITH optimal_prices AS (
  SELECT
    optimization_group,
    optimization_size,
    SUM(optimization_value * selector) AS interpolated_value,
    ANY_VALUE(opt_name) AS opt_name,
  FROM
    `{{ optimization_selector_table }}`
  GROUP BY
    optimization_group,
    optimization_size
),
selected_prices AS (
  SELECT
    -- handle null value to 0 in case the gross_red_price is greater than interpolated_price
    IFNULL(
      FIRST_VALUE(
        IF(
          ROUND(optimization_value, 2) <= ROUND(interpolated_value, 2),
          ROUND(optimization_value, 2),
          NULL
        ) IGNORE NULLS
      ) OVER(price_points ORDER BY optimization_value DESC)
      = ROUND(optimization_value, 2),
      FALSE
    ) AS selector,
    market,
    channel,
    product_id,
    predicted_sales_before_returns,
    predicted_revenue,
    predicted_profit,
    gross_red_price,
    avg_gross_black_price,
    opt_name,
    scope_bits,
  FROM
    optimal_prices
    JOIN `{{ filtered_table }}` USING (optimization_group, optimization_size)
  WINDOW
    price_points AS (
      PARTITION BY optimization_group, CAST(optimization_size AS STRING)
    )
)

SELECT
  CAST(selector AS INT64) AS selector,
  * EXCEPT(selector),
FROM
  selected_prices
