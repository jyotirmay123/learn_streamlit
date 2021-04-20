SELECT
  *
FROM (
  SELECT
    market,
    channel,
    product_id,
    nth_most_recent_price_change,
    MAX(nth_most_recent_price_change)OVER(PARTITION BY market, channel, product_id) AS nr_price_changes,
    gross_red_price AS gross_red_price,
    avg_sales_before_returns AS sales_before_returns,
    avg_revenue AS revenue,
    avg_profit AS profit,
    DATE_DIFF(DATE(active_till), DATE(active_since), DAY) AS num_days,
  FROM
{{ table }}
  WHERE
    DATE_DIFF(DATE(active_till), DATE(active_since), DAY) > 5
    AND DATE(active_since) > DATE_SUB(CURRENT_DATE(), INTERVAL 280 DAY)
)
WHERE nr_price_changes > {{ nr_price_changes }}
