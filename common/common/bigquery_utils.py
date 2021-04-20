import typing

import pandas as pd
from google.cloud import bigquery as bq, bigquery_storage as bqs


def load_dataframe(
    dataframe: pd.DataFrame,
    destination_table: str,
    overwrite_if_exists: bool = True,
    bq_client: bq.Client = None,
):
    """Load pandas Dataframe into `destination_table`, overwriting existing tables by default."""
    write_disposition = (
        bq.job.WriteDisposition.WRITE_TRUNCATE
        if overwrite_if_exists
        else bq.job.WriteDisposition.WRITE_APPEND
    )
    if bq_client is None:
        bq_client = bq.Client()
    job = bq_client.load_table_from_dataframe(
        dataframe,
        destination_table,
        job_config=bq.LoadJobConfig(
            create_disposition=bq.job.CreateDisposition.CREATE_IF_NEEDED,
            write_disposition=write_disposition,
        ),
    ).result()

    print("Loaded {} rows into `{}`.".format(job.output_rows, destination_table))


def table_to_dataframe(
    bqrc: bqs.BigQueryReadClient,
    table: str,
    *,
    selected_fields: typing.Sequence[str] = [],
    row_restriction: str = None,
) -> pd.DataFrame:
    """
    Fetch a table from BigQuery and return it as pandas DataFrame.

    Parameters:
      table: Fully Qualified name of table to read
      selected_fields: see :attr:`google.cloud.bigquery_storage.types.ReadSession.TableReadOptions.selected_fields`
      row_restriction: see :attr:`google.cloud.bigquery_storage.types.ReadSession.TableReadOptions.row_restriction`

    See Also:
      :meth:`bq_reader`
    """
    reader, session = bq_reader(
        bqrc, table, selected_fields=selected_fields, row_restriction=row_restriction
    )
    return reader.to_dataframe(session)


def bq_reader(
    bqrc: bqs.BigQueryReadClient,
    table,
    *,
    selected_fields: typing.Sequence[str] = [],
    row_restriction: str = None,
) -> bqs.types.ReadSession:
    """
    Create a BigQuery Storage API Read Session for the given `table`

    Parameters:
      table: Fully Qualified name of table to read
      selected_fields: see :attr:`google.cloud.bigquery_storage.types.ReadSession.TableReadOptions.selected_fields`
      row_restriction: see :attr:`google.cloud.bigquery_storage.types.ReadSession.TableReadOptions.row_restriction`
    """
    table = bq.table.TableReference.from_string(table)
    sess_req = bqs.types.ReadSession()
    sess_req.table = (
        f"projects/{table.project}/datasets/{table.dataset_id}/tables/{table.table_id}"
    )
    sess_req.data_format = bqs.types.DataFormat.ARROW
    if selected_fields:
        sess_req.read_options.selected_fields.extend(selected_fields)
    if row_restriction is not None:
        sess_req.read_options.row_restriction = row_restriction
    sess = bqrc.create_read_session(
        parent=f"projects/{table.project}", read_session=sess_req, max_stream_count=1
    )
    reader = bqrc.read_rows(sess.streams[0].name)
    return reader, sess


def parse_schema(schema: str) -> typing.List[bq.SchemaField]:
    """Parses BigQuery's short (CLI) schema format.
    A comma-separated list of fields in the form name[:type], where type defaults to STRING.

    Examples:
      parse_schema('market,channel,product_id:INT64')
    """
    ret = []
    for col in schema.split(','):
        name, type_ = (col.split(':', 1) + ['STRING'])[:2]
        ret.append(bq.SchemaField(name, type_))
    return ret
