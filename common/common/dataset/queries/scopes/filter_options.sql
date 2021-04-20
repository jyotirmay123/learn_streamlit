WITH
agg AS (
  SELECT
    {% for col in columns %}
    ARRAY_AGG(DISTINCT {{ col }} IGNORE NULLS LIMIT {{ limit }}) AS {{ col }},
    {% endfor %}
  FROM
    `{{ table }}`
)

SELECT
  {% for col in columns %}
  IF(ARRAY_LENGTH({{ col}}) >= {{ limit }}, NULL, {{ col }}) AS {{ col }},
  {% endfor %}
FROM agg
