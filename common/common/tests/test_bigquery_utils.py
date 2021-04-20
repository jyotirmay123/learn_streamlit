import google.cloud.bigquery as bq


def test_parse_schema():
    from ..bigquery_utils import parse_schema

    expected = [bq.SchemaField('market', 'STRING'), bq.SchemaField('channel', 'STRING')]
    assert parse_schema('market,channel') == expected
    expected.append(bq.SchemaField('product_id', 'INT64'))
    assert parse_schema('market,channel,product_id:INT64') == expected
    expected.append(bq.SchemaField('value', 'FLOAT64'))
    assert (
        parse_schema('market,channel:STRING,product_id:INT64,value:FLOAT64') == expected
    )
