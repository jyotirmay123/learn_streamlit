{# replicate each row for each individual scope_bit that is included in scope_bits #}
    INNER JOIN UNNEST([
    {%- set bit_sep = joiner(", ") %}
    {%- for scope in scopes %}
      {{ bit_sep() }}STRUCT({{ (1).__lshift__(loop.index0) }} AS bit, '{{ scope.name }}' AS name)
    {%- endfor %}
    ]) AS scope ON scope_bits & scope.bit != 0
