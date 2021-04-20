SELECT
  market,
  channel,
  product_id,
  gross_red_price,
  SUM(predicted_sales_before_returns) AS sales_before_returns,
  SUM(predicted_returns) AS returns,
  SUM(predicted_revenue) AS revenue,
  SUM(predicted_profit) AS profit,
FROM
  {% if '$' in forecast_table %}
    {% set forecast_table, partition_date = forecast_table.split('$') %}
  {% endif %}
  {{ forecast_table }}
  {% if partition_date %}
    WHERE DATE(_PARTITIONTIME) = PARSE_DATE('%Y%m%d', "{{ partition_date }}")
  {% endif %}
GROUP BY
  market,
  channel,
  product_id,
  gross_red_price
