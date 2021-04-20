{# join with the generated filter subqueries of gen_filter.sql #}
{%- set entities = rules if use_rules else scopes -%}
{%- for entity in entities %}
  {%- if entity.filters | length %}
    LEFT JOIN scope_{{ loop.index0 }} USING(
    {%- set kpi_sep = joiner(", ") -%}
    {%- for item in entity.filters -%}
     {{ kpi_sep() }}{{ item.kpi }}
    {%- endfor -%}
    )
  {%- else %}
    CROSS JOIN scope_{{ loop.index0 }}
  {%- endif %}
{%- endfor %}
