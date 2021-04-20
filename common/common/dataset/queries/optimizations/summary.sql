-- v2.0
WITH
{%- if non_overlapping_attributes %}
product_attributes AS (
  SELECT
    product_id,
    {%- for col in non_overlapping_attributes %}
    {{ col }},
    {%- endfor %}
  FROM
    data.product_attributes
),
{%- endif %}
scoped_optimizations AS (
  {%- set use_rules = False %}
  {%- include "./scopes/gen_scopes.sql" %}
  SELECT
    -- combine scope_bits for all scopes
    (
    {%- set scope_sep = joiner("| ") %}
    {%- for scope in scopes %}
      {{ scope_sep() }}IFNULL(scope_{{ loop.index0 }}.scope_bit, 0)
    {%- endfor %}
    ) AS scope_bits,
    -- copy over all optimization columns as opaque STRUCT
    (SELECT AS STRUCT opt.* EXCEPT(snapshot_id)) AS opt,
  FROM
    `{{ table }}` AS opt
    {%- if non_overlapping_attributes %}
    LEFT JOIN product_attributes USING(product_id)
    {%- endif %}
    {%- include "./scopes/join_scopes.sql" %}
  WHERE
    snapshot_id = {{ snapshot_id }}
    -- only keep products included in at least one scope
    AND (
    {%- set scope_sep = joiner("OR ") %}
    {%- for scope in scopes %}
      {{ scope_sep() }}scope_{{ loop.index0 }}.scope_bit IS NOT NULL
    {%- endfor %}
    )
),
aggregated AS (
  SELECT
    scope.bit AS scope_bit,
    ANY_VALUE(scope.name) AS scope,
    SUM(opt.predicted_sales_before_returns) AS predicted_sales_before_returns,
    SUM(opt.predicted_revenue) AS predicted_revenue,
    SUM(opt.predicted_profit) AS predicted_profit,
    SAFE_DIVIDE(SUM(opt.predicted_profit), SUM(opt.predicted_revenue)) AS predicted_profit_margin,
    SAFE_DIVIDE(SUM(opt.predicted_revenue * opt.price_change), SUM(opt.predicted_revenue)) AS wavg_price_change,
    COUNT(*) AS count,
    -- include top/bottom opt download rows of each column to find outliers
    ARRAY_CONCAT(
      {%- set arg_sep = joiner(', ') %}
      {%- for col in table_columns %}
      {{ arg_sep() }}ARRAY_AGG(opt ORDER BY opt.{{ col }} ASC LIMIT 50)
      {{ arg_sep() }}ARRAY_AGG(opt ORDER BY opt.{{ col }} DESC LIMIT 50)
      {%- endfor %}
    ) AS extremes,
    -- gather all KPI values to compute value counts/histograms
    STRUCT(
      {%- set arg_sep = joiner(', ') %}
      {%- for col in table_columns %}
      {#- TODO: limit array size? #}
      {{ arg_sep() }}ARRAY_AGG(opt.{{ col }}) AS {{ col }}
      {%- endfor %}
    ) AS histograms,
  FROM
    scoped_optimizations
    -- flatten product_optima of each scope
    {%- include "./scopes/split_scopes.sql" %}
  GROUP BY
    scope_bit
)
SELECT
  * EXCEPT(scope_bit, extremes, histograms),
  -- dedup top/bottom rows opt download rows
  ARRAY((
    SELECT
      DISTINCT AS STRUCT *
    FROM
      UNNEST(extremes)
  )) AS extremes,
  -- Calculate value counts to compress histogram data. Use a sub-query instead
  -- of a temporary function to allow query result caching, would alternatively
  -- also work with a persistent fx.HISTOGRAM function.
  STRUCT(
    {%- set arg_sep = joiner(', ') %}
    {%- for col in table_columns %}
    {{ arg_sep() }}ARRAY(SELECT AS STRUCT value, COUNT(*) AS cnt FROM UNNEST(histograms.{{ col }}) AS value GROUP BY value) AS {{ col }}
    {%- endfor %}
  ) AS histograms,
FROM
  aggregated
