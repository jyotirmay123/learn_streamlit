import typing

import pandas as pd
import streamlit as st
from google.cloud import resource_manager, bigquery as bq, bigquery_storage as bqs

from common.common.bigquery_utils import table_to_dataframe


_project_id = None


def select_gcp_project(default: str = None) -> str:
    """Select GCP project"""
    global _project_id
    rmc = resource_manager.Client()
    projects = {p.project_id: p.name for p in rmc.list_projects()}
    _project_id = st.selectbox(
        'Select project',
        options=[None] + list(projects.keys()),
        format_func=lambda pid: f"{projects[pid]} ({pid})" if pid else 'None',
    )
    if _project_id is None:
        st.stop()
        return ''  # for mypy
    return _project_id


class ProgressBar:
    """Streamlit-based progress bar for BigQuery client progress"""

    def __init__(self, bar):
        self.bar = bar
        self.total = 0
        self.num = 0

    def update(self, num: int):
        self.num += num
        self.bar.progress(int(100 * self.num / self.total) if self.total else 0)

    def close(self):
        self.bar.progress(100)


@st.cache(persist=True, suppress_st_warning=True)
def bq_query(
    gcp_project_id: str,
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
    st.code(query, language='sql')
    job_config = bq.job.QueryJobConfig(query_parameters=params)
    # use storage api for faster transfer
    rows = bq.Client(gcp_project_id).query(query, job_config=job_config).result()
    rows._get_progress_bar = lambda _: ProgressBar(st.progress(0))
    return rows.to_dataframe(create_bqstorage_client=True)


@st.cache(persist=True, suppress_st_warning=True)
def bq_download(
    gcp_project_id: str,
    table: str,
    *,
    selected_fields: typing.Sequence[str] = [],
    row_restriction: str = None,
) -> pd.DataFrame:
    """Run possibly parameterized query against BigQuery and return the result as a dataframe."""
    fqtn = f"{gcp_project_id}.{table}"
    st.write(f"Downloading {fqtn}")
    # TODO: progressbar
    return table_to_dataframe(
        bqs.BigQueryReadClient(),
        fqtn,
        selected_fields=selected_fields,
        row_restriction=row_restriction,
    )


def bq_upload(gcp_project_id: str, destination: str, df: pd.DataFrame,) -> int:
    """Upload df to destination."""
    return (
        bq.Client(gcp_project_id)
        .load_table_from_dataframe(
            df,
            destination,
            job_config=bq.job.LoadJobConfig(
                write_disposition=bq.job.WriteDisposition.WRITE_TRUNCATE,
                create_disposition=bq.job.CreateDisposition.CREATE_IF_NEEDED,
            ),
        )
        .result()
        .output_rows
    )


@st.cache()
def bq_schema(project_id: str, table: str) -> typing.Dict[str, bq.SchemaField]:
    return {
        field.name: field for field in bq.Client(project_id).get_table(table).schema
    }


@st.cache()
def bq_list_tables(
    project_id: str, dataset: str, *, prefix: str = None
) -> typing.List[str]:
    return [
        f"{dataset}.{tableItem.reference.table_id}"
        for tableItem in sorted(
            bq.Client(project_id).list_tables(dataset),
            key=lambda tableItem: tableItem.created,
            reverse=True,
        )
        if prefix is None or tableItem.reference.table_id.startswith(prefix)
    ]


@st.cache()
def bq_list_partitions(project_id: str, table: str) -> typing.List[str]:
    return [
        f"{table}${partition_date.strftime('%Y%m%d')}"
        for partition_date in sorted(
            map(
                lambda row: row['date'],
                bq.Client(project_id)
                .query(f"SELECT DISTINCT _PARTITIONDATE AS date FROM `{table}`")
                .result(),
            ),
            reverse=True,
        )
    ]
