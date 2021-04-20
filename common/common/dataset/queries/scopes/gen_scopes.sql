{#- generate all row combinations included by the filters of each scope/rule #}
{%- set query_sep = joiner(",") %}
{%- set entities = rules if use_rules else scopes -%}
{%- if entities | length %}
  WITH
{%- endif -%}
{%- for entity in entities %}
  {{- query_sep() }}
  scope_{{ loop.index0 }} AS (
    {%- set filters = entity.filters %}
    {%- if filters | length %}
    -- use DISTINCT to be robust against duplicates in filters
    SELECT DISTINCT
    {%- else %}
    SELECT
    {%- endif %}
      {%- if use_rules %}
      1 AS dummy_col,
      {%- else %}
      {#- workaround for missing bitwise operators https://github.com/pallets/jinja/issues/249#issue-16602984 #}
      {{ (1).__lshift__(loop.index0) }} AS scope_bit,
      {%- endif %}
      {%- for item in filters %}
        {{ item.kpi }},
      {%- endfor %}
    {%- if filters | length %}
    FROM
      {%- set filter_sep = joiner(" CROSS JOIN ") %}
      {%- for item in filters %}
      {{ filter_sep() }}UNNEST({{ item.options|list }}) AS {{ item.kpi }}
      {%- endfor %}
    {%- endif %}
  )
{%- endfor %}
