"""This package provides a standardized coding API for client-specific BigQuery data.

The main purpose is to facilitate and standardizing data querying for the
backend and optimizer.
"""

from .bigquery import bq, bqs, bq_client, bq_read_client  # noqa: F401
from .templating import render_sql_from_file  # noqa: F401
