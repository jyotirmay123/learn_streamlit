import typing

import pandas as pd
from google.cloud import bigquery as bq, bigquery_storage_v1 as bqs

_bq_client: bq.Client = None
_bq_read_client: bqs.BigQueryReadClient = None


def bq_client(gcp_project: str = None) -> bq.Client:
    """Cached BigQuery client.

    You can explicitly initialize the cache with a `gcp_project` that
    will be passed to :class:`bq.Client <google.cloud.bigquery.client.Client>`.
    """
    global _bq_client
    if _bq_client is None:
        _bq_client = bq.Client(gcp_project)
    return _bq_client


def bq_read_client() -> bqs.client.BigQueryReadClient:
    """Cached BigQuery Storage API client."""
    global _bq_read_client
    if _bq_read_client is None:
        _bq_read_client = bqs.BigQueryReadClient()
    return _bq_read_client


def bq_query(
    project_id: str,
    query: str,
    params: typing.List[
        typing.Union[
            bq.query.ArrayQueryParameter,
            bq.query.ScalarQueryParameter,
            bq.query.StructQueryParameter,
        ]
    ] = [],
) -> pd.DataFrame:
    """Run possibly parameterized query against BigQuery and return the result as a dataframe."""
    job_config = bq.job.QueryJobConfig(query_parameters=params)
    # use storage api for faster transfer
    rows = bq.Client(project_id).query(query, job_config=job_config).result()
    return rows.to_dataframe(create_bqstorage_client=True)
