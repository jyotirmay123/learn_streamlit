CREATE TEMPORARY FUNCTION FILTER(value ANY TYPE, selected_values ANY TYPE) AS (
  ARRAY_LENGTH(selected_values) = 0
  OR value IN UNNEST(selected_values)
);

WITH latest_partitions AS (
  SELECT
    _PARTITIONDATE AS date,
  FROM  production.forecast
  WHERE _PARTITIONDATE > DATE_SUB(CURRENT_DATE(), INTERVAL 31 DAY)
),
scopes_info AS (
  {%- set scope_sep = joiner(" UNION ALL") %}
  {% for scope in scopes %}
  {{- scope_sep() }}
  SELECT
    {{ scope.id }} AS id,
    COUNT(DISTINCT CONCAT(market, '-', channel, '-', product_id)) AS cnt,
  FROM
      production.forecast
      LEFT JOIN data.product_attributes USING(product_id)
  WHERE
      _PARTITIONDATE = (SELECT MAX(date) FROM latest_partitions)
      {%- for item in scope.filters %}
      {# skip if there is no any values in the options to avoid BigQuery error #}
      {% if item.options|length > 0 %}
      AND FILTER({{ item.kpi }}, {{ item.options|list }})
      {% endif %}
      {%- endfor %}
  {% endfor %}
)

SELECT
  *
FROM
  scopes_info
