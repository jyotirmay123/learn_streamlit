/*
  y = m * x + c

  m = (y₁ - y₀) / (x₁ - x₀) = Δy / Δx
  c = y₀ - mx₀ = y₁ - mx₁

  y' = m * x' + c
  y' = m * x' + y₀ - mx₀
  y' = Δy / Δx (x' - x₀) + y₀
  y' = (x' - x₀) / Δx * Δy + y₀
  y' = rel_delta_x * Δy + y₀
 */
CREATE TEMPORARY FUNCTION INTERPOLATE2(
  optimization_group STRING,
  optimization_size FLOAT64,
  rel_delta_x FLOAT64,
  val0 ANY TYPE,
  val1 ANY TYPE
) AS (
  STRUCT(
    rel_delta_x * (val1.price - val0.price) + val0.price AS price,
    optimization_group,
    optimization_size,
    rel_delta_x * (val1.optimization_value - val0.optimization_value) + val0.optimization_value AS optimization_value,
    rel_delta_x * (val1.predicted_sales_before_returns - val0.predicted_sales_before_returns) + val0.predicted_sales_before_returns AS predicted_sales_before_returns,
    rel_delta_x * (val1.predicted_revenue - val0.predicted_revenue) + val0.predicted_revenue AS predicted_revenue,
    rel_delta_x * (val1.predicted_profit - val0.predicted_profit) + val0.predicted_profit AS predicted_profit
  )
);

CREATE TEMPORARY FUNCTION INTERPOLATE(
  price FLOAT64,
  a STRUCT<
    price FLOAT64,
    optimization_group STRING,
    optimization_size FLOAT64,
    optimization_value FLOAT64,
    predicted_sales_before_returns FLOAT64,
    predicted_revenue FLOAT64,
    predicted_profit FLOAT64
  >,
  b STRUCT<
    price FLOAT64,
    optimization_group STRING,
    optimization_size FLOAT64,
    optimization_value FLOAT64,
    predicted_sales_before_returns FLOAT64,
    predicted_revenue FLOAT64,
    predicted_profit FLOAT64
  >
) AS (
  IF(
    -- only inter-/extrapolate if we have two adjacent prices
    a IS NULL OR b IS NULL,
    NULL,
    INTERPOLATE2(
      a.optimization_group,
      a.optimization_size,
      (price - a.price) / (b.price - a.price),
      a,
      b
    )
  )
);

####################### FUNCTIONS ABOVE #######################

{% if forecast_date | length %}
  {% set forecast_date_stmt = "_PARTITIONDATE='{}' AND".format(forecast_date) %}
{% else %}
  {% set forecast_date_stmt = None %}
{% endif %}

WITH

-- Step 1: selecting products within the scopes and add cur_gross_red_price
scoped_forecast AS (
  {%- set use_rules = False %}
  {%- include "./scopes/gen_scopes.sql" %}
  SELECT
    market,
    channel,
    product_id,
    forecast.gross_red_price AS gross_red_price,
    ANY_VALUE((
      SELECT AS STRUCT price_periods.gross_red_price AS cur_gross_red_price
    )) AS price_periods,
    -- Combine scope_bits for all scopes
    ANY_VALUE(
    {%- set scope_sep = joiner("| ") %}
    {%- for scope in scopes %}
      {{ scope_sep() }}IFNULL(scope_{{ loop.index0 }}.scope_bit, 0)
    {%- endfor %}
    ) AS scope_bits,
    STRUCT(
      forecast.gross_red_price AS gross_red_price,
      -- workaround to avoid product_attributes has optimization_* columns (e.g., munich-01)
      ANY_VALUE(forecast.optimization_group) AS optimization_group,
      ANY_VALUE(forecast.optimization_size) AS optimization_size,
      ANY_VALUE(forecast.optimization_value) AS optimization_value,
      SUM(predicted_sales_before_returns) AS predicted_sales_before_returns,
      SUM(predicted_revenue) AS predicted_revenue,
      SUM(predicted_profit) AS predicted_profit
    ) AS predictions,
    -- include non-standard forecast columns as additional KPIs for custom rule predicates (e.g. is_sale_price)
    -- but keep one standard forecast column (market) to avoid empty struct
    ANY_VALUE((
      SELECT AS STRUCT forecast.* EXCEPT(
        channel, product_id, date, gross_red_price,
        optimization_group, optimization_size, optimization_value,
        predicted_sales_before_returns, predicted_returns, predicted_revenue, predicted_profit, predicted_elasticity,
        predicted_avg_voucher_spending, predicted_avg_inbound_cost, predicted_avg_outbound_cost
        -- include predicted_avg_purchase_price since might be used in the rule
        -- predicted_avg_purchase_price
      )
    )) AS forecast,
  FROM
    `{{ forecast_table }}` AS forecast
    LEFT JOIN data.product_attributes AS product_attributes USING(product_id)
    LEFT JOIN data.price_periods AS price_periods USING(market, channel, product_id)
    {%- include "./scopes/join_scopes.sql" %}
  WHERE
    {{ forecast_date_stmt or '' }}
    date BETWEEN '{{ start_date }}' AND '{{ end_date }}'
    AND IFNULL(nth_most_recent_price_change, 1) = 1
    -- only keep products included in at least one scope
    AND (
    {%- set scope_sep = joiner("OR ") %}
    {%- for scope in scopes %}
      {{ scope_sep() }}scope_{{ loop.index0 }}.scope_bit IS NOT NULL
    {%- endfor %}
    )
  GROUP BY
    market,
    channel,
    product_id,
    gross_red_price
),

-- Step 2: Adding current price or specified price (e.g. rule: price = 13.65)
additional_prices AS (
  WITH
  {%- if require_interpolated_prices %}
  sythesized_prices AS (
    {%- set use_rules = True %}
    {%- include "./scopes/gen_scopes.sql" %}
    SELECT
      DISTINCT market, channel, product_id, gross_red_price,
    FROM
      scoped_forecast
      LEFT JOIN data.product_attributes AS product_attributes USING(product_id)
      {%- include "./scopes/join_scopes.sql" %}
      -- Interpolated price rules with overlapping scopes can lead to multiple entries here and will
      -- filter each other out later during the filtering stage, thus keeping the current price for
      -- such cases. This is in line with how we handle other contradictory rules.
      CROSS JOIN UNNEST([
        {%- set rule_sep = joiner(",") %}
        {%- for rule in rules %}
          {%- if rule.interpolated_price %}
        {{- rule_sep() }}
        IF(scope_{{ loop.index0 }}.dummy_col IS NULL, NULL, ({{ rule.interpolated_price }}))
          {%- endif %}
        {%- endfor %}
      ]) AS gross_red_price
    WHERE
      gross_red_price IS NOT NULL
  ),
  {% endif %}

  current_prices AS (
    SELECT
      market,
      channel,
      product_id,
      gross_red_price,
      TRUE AS is_current_price,
    FROM
      data.price_periods
    WHERE
      nth_most_recent_price_change = 1
  ),

  scoped_products AS (
    SELECT DISTINCT
      market,
      channel,
      product_id,
    FROM
      scoped_forecast
  )

  SELECT
    *
  FROM
    current_prices
    {% if require_interpolated_prices %}
    FULL JOIN sythesized_prices USING(market, channel, product_id, gross_red_price)
    {% endif %}
    -- restrict to products included in forecast
    INNER JOIN scoped_products USING(market, channel, product_id)
),

-- Step 3: Adding additional price points and interpolating them from existing predictions
full_forecast AS (
  SELECT
    market, channel, product_id,
    gross_red_price,
    IFNULL(is_current_price, FALSE) AS is_current_price,
    scoped_forecast.gross_red_price IS NULL AS is_interpolated,
    -- fill gaps from adding rows with synthesized prices above in Step 3
    IFNULL(
      scope_bits,
      FIRST_VALUE(scope_bits IGNORE NULLS) OVER(adjacent)
    ) AS scope_bits,
    IFNULL(
      price_periods,
      FIRST_VALUE(price_periods IGNORE NULLS) OVER(adjacent)
    ) AS price_periods,
    IFNULL(
      forecast,
      FIRST_VALUE(forecast IGNORE NULLS) OVER(adjacent)
    ) AS forecast,
    COALESCE(
      predictions,
      INTERPOLATE(
        gross_red_price,
        IFNULL(
          -- use preceding smaller price
          LAG(predictions) OVER(closest),
          -- or 2nd next following higher price as fallback
          LEAD(predictions, 2) OVER(closest)
        ),
        IFNULL(
          -- use following higher price
          LEAD(predictions) OVER(closest),
          -- or 2nd next preceding smaller price as fallback
          LAG(predictions, 2) OVER(closest)
        )
      ),
      FIRST_VALUE(predictions IGNORE NULLS) OVER(adjacent)
    ) AS predictions,
  FROM
    -- use FULL JOIN to add all prices while preventing duplicates, this will leave NULL values for non-overlapping columns
    scoped_forecast
    FULL JOIN additional_prices USING(market, channel, product_id, gross_red_price)
  WINDOW
    closest AS (PARTITION BY market, channel, product_id ORDER BY gross_red_price ASC),
    adjacent AS (closest ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING)
),

-- Step 4: Apply the rules and tag the point options if they're out
rules_pass_1 AS (
  {%- set use_rules = True %}
  {%- include "./scopes/gen_scopes.sql" %}
  SELECT
    market,
    channel,
    product_id,
    scope_bits,
    ff.gross_red_price,
    is_current_price,
    -- unpack predictions struct into columns
    predictions.optimization_group AS optimization_group,
    predictions.optimization_size AS optimization_size,
    predictions.optimization_value AS optimization_value,
    predictions.predicted_sales_before_returns AS predicted_sales_before_returns,
    predictions.predicted_revenue AS predicted_revenue,
    predictions.predicted_profit AS predicted_profit,
    forecast.gross_black_price AS avg_gross_black_price,
    -- don't introduce interpolated current prices if we have other price valid points
    (is_current_price AND is_interpolated)
    {%- if rules | length %}
    -- apply the rules and mark if the price is out
    OR NOT (
      {%- set rule_sep = joiner("AND") %}
      {%- for rule in rules %}
      {{ rule_sep() }}
      (
        scope_{{ loop.index0 }}.dummy_col IS NULL
        OR IFNULL({{ rule.predicate }}, TRUE)
      )
      {%- endfor %}
    )
    {%- endif %}
    AS is_filtered_out,
  FROM
    full_forecast AS ff
    LEFT JOIN data.product_attributes AS product_attributes USING(product_id)
    {%- include "./scopes/join_scopes.sql" %}
),

-- Step 5: Add a column to indicate if all price options are out
rules_pass_2 AS (
  SELECT
    *,
    -- a checker if all the prices have been filtered out
    LOGICAL_AND(is_filtered_out) OVER(products) AS is_all_filtered_out,
  FROM
    rules_pass_1
  WINDOW
    products AS (PARTITION BY market, channel, product_id),
    adjacent AS (products ORDER BY gross_red_price ASC ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING)
),

-- Final selection of price options after all rules are applied
price_options_after_rules AS (
  SELECT
    * EXCEPT (is_all_filtered_out, is_filtered_out),
    -- Case where we have valid price options from the forecast
    is_filtered_out
    -- Case where all price points in forecast where kicked out by rules in which we keep the (interpolated) current price
    AND NOT (
      is_all_filtered_out
      AND is_current_price
    ) AS is_filtered_out,
  FROM
    rules_pass_2
),

-- Prepare preview for front-end
preview_optima AS (
  WITH product_optima AS (
    SELECT
      market,
      channel,
      product_id,
      ANY_VALUE(scope_bits) AS scope_bits,
      ARRAY_AGG(IF(is_filtered_out, NULL, STRUCT(predicted_revenue, predicted_profit)) IGNORE NULLS ORDER BY  predicted_profit DESC LIMIT 1)[OFFSET(0)] AS  max_predicted_profit,
      ARRAY_AGG(IF(is_filtered_out, NULL, STRUCT(predicted_revenue, predicted_profit)) IGNORE NULLS ORDER BY predicted_revenue DESC LIMIT 1)[OFFSET(0)] AS max_predicted_revenue,
      ARRAY_AGG(IF(is_filtered_out, NULL, STRUCT(predicted_revenue, predicted_profit)) IGNORE NULLS ORDER BY predicted_revenue  ASC LIMIT 1)[OFFSET(0)] AS min_predicted_revenue,
      /*
        Disable hard error check for current prices to not break preview for few edge cases
        (e.g. changed assortment between older forecast and more recent current prices).
      IF(
        SUM(IF(is_current_price, 1, 0)) = 1,
        ARRAY_AGG(IF(is_current_price, STRUCT(predicted_revenue, predicted_profit), NULL) IGNORE NULLS LIMIT 1)[OFFSET(0)],
        ERROR(FORMAT('Found an unexpected number %t of current prices for %t %t %t', SUM(IF(is_current_price, 1, 0)), market, channel, product_id))
      ) AS current_prices,
      */
      ARRAY_AGG(IF(is_current_price, STRUCT(predicted_revenue, predicted_profit), NULL) IGNORE NULLS LIMIT 1)[OFFSET(0)] AS current_prices,
    FROM
      price_options_after_rules
    GROUP BY
      market,
      channel,
      product_id
  )
  SELECT
    scope.bit AS scope_bit,
    STRUCT(ROUND(SUM(max_predicted_profit.predicted_profit), 2) AS profit, ROUND(SUM(max_predicted_profit.predicted_revenue), 2) AS revenue) AS  max_predicted_profit,
    STRUCT(ROUND(SUM(max_predicted_revenue.predicted_profit), 2) AS profit, ROUND(SUM(max_predicted_revenue.predicted_revenue), 2) AS revenue) AS max_predicted_revenue,
    STRUCT(ROUND(SUM(min_predicted_revenue.predicted_profit), 2) AS profit, ROUND(SUM(min_predicted_revenue.predicted_revenue), 2) AS revenue) AS min_predicted_revenue,
    STRUCT(ROUND(SUM(current_prices.predicted_profit), 2) AS profit, ROUND(SUM(current_prices.predicted_revenue), 2) AS revenue) AS current_prices,
  FROM
    product_optima
    -- flatten product_optima of each scope
    {%- include "./scopes/split_scopes.sql" %}
  GROUP BY
    scope.bit
)

-- Select either the preview OR the price options after rules application
{% if is_preview %}
SELECT
  *
FROM
  preview_optima
{% else %}
SELECT
  optimization_group,
  optimization_size,
  optimization_value,
  market, channel, product_id,
  scope_bits,
  predicted_sales_before_returns,
  predicted_revenue,
  predicted_profit,
  gross_red_price,
  avg_gross_black_price,
FROM
  price_options_after_rules
WHERE
  NOT is_filtered_out
{% endif %}
