# Copyright © 2019 7Learnings GmbH
"""Dataset utilities"""
import enum
import io
import os
import typing

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import tensorflow as tf
from google.cloud import bigquery_storage as bqs

from common.bigquery_utils import bq_reader


_ARROW_TO_TF = {
    pyarrow.string(): tf.string,
    pyarrow.float64(): tf.float32,
    pyarrow.int64(): tf.int32,
    pyarrow.bool_(): tf.bool,
    pyarrow.timestamp('us', tz='UTC'): None,
    pyarrow.date32(): tf.string,
}


def _arrow_schema_to_tf(
    arrow_schema: pyarrow.Schema, time_series_length: int,
) -> typing.Dict[str, tf.io.FixedLenFeature]:
    return {
        name: tf.io.FixedLenFeature(
            [time_series_length, 1]
            if isinstance(pya_type, pyarrow.lib.ListType)
            else [1],
            dtype=_ARROW_TO_TF[
                pya_type.value_type
                if isinstance(pya_type, pyarrow.lib.ListType)
                else pya_type
            ],
        )
        for name, pya_type in zip(arrow_schema.names, arrow_schema.types)
    }


@enum.unique
class Split(enum.Enum):
    """Enum to select dataset train or eval dataset splits"""

    train = 'train'
    eval = 'eval'
    predict = 'predict'


def table_to_parquet(
    bqrc: bqs.BigQueryReadClient,
    table: str,
    data_dir: str,
    row_restriction: typing.Optional[str] = None,
) -> None:
    """
    Fetch a table from BigQuery and store it in parquet format in data_dir.

    Parameters:
      row_restriction: see :attr:`google.cloud.bigquery_storage.types.ReadSession.TableReadOptions.row_restriction`
    """
    reader, session = bq_reader(bqrc, table, row_restriction=row_restriction)
    pqf: pq.ParquetWriter = None
    buf = io.BytesIO()
    try:
        for page in reader.rows(session).pages:
            batch = page.to_arrow()
            if pqf is None:
                pqf = pq.ParquetWriter(buf, batch.schema)
            pqf.write_table(pyarrow.Table.from_batches([batch]))
    finally:
        if pqf is not None:
            pqf.close()

    try:
        tf.io.write_file(os.path.join(data_dir, f"{table}.parquet"), buf.getvalue())
    finally:
        buf.close()


def load_datasets(
    data_dir: str,
    table: str,
    time_series_length: int,
    input_label_cols: typing.List[str],
    data_split_col: str,
    data_shuffle_col: str,
    exclude_columns: typing.List[str] = [],
    include_columns: typing.List[str] = None,
) -> typing.Tuple[
    typing.Dict[Split, tf.data.Dataset], typing.Dict[str, tf.io.FixedLenFeature]
]:
    path = os.path.join(data_dir, f"{table}.parquet")
    if not tf.io.gfile.exists(path):
        raise ValueError(
            f"Did not find expected file {path}, call table_to_parquet first."
        )

    return _parquet_to_tf_dataset(
        path,
        time_series_length,
        input_label_cols,
        data_split_col,
        data_shuffle_col,
        exclude_columns,
        include_columns,
    )


def build_vocabularies(
    data: tf.data.Dataset, cat_features: typing.List[str]
) -> typing.Dict[str, typing.Any]:
    counts: dict = {cat: {} for cat in cat_features}
    for i in range(0, len(data), 1024):
        X, Y = data[i : i + 1024]
        for cat in cat_features:
            temp = X[cat].numpy().ravel()
            cat_counts = counts[cat]
            for val in temp:
                cat_counts[val] = cat_counts.get(val, 0) + 1
    counts = {
        cat: pd.Series(cat_counts).sort_values(ascending=False)
        for cat, cat_counts in counts.items()
    }
    vocabularies = {}
    for col in cat_features:
        serie = counts[col]
        serie = (serie / serie.sum()) * 100
        serie = serie[serie > 1]
        vocabularies[col] = {
            'labels': list(serie.index),
            'counts_normalized': serie.tolist(),
        }
    return vocabularies


class Dataset:
    def __init__(
        self,
        X: typing.Dict[str, tf.Tensor],
        Y: typing.Union[tf.Tensor, typing.Dict[str, tf.Tensor]],
    ):
        self.X = X
        self.Y = Y
        self.length = len(next(iter(X.values())))

    def __len__(self):
        return self.length

    def __getitem__(
        self, s: slice
    ) -> typing.Tuple[
        typing.Dict[str, tf.Tensor],
        typing.Union[tf.Tensor, typing.Dict[str, tf.Tensor]],
    ]:
        Xbatch = {name: t[s] for name, t in self.X.items()}
        if isinstance(self.Y, dict):
            Ybatch = {name: t[s] for name, t in self.Y.items()}
        else:
            Ybatch = self.Y[s]
        return (Xbatch, Ybatch)


def _pandas_to_tf_dataset(
    df: pd.DataFrame, feature_specs, input_label_cols,
):
    tensors = {
        name: np.expand_dims(df[name].values, -1)
        if len(feat.shape) == 1
        else np.expand_dims(np.vstack(df[name].values), -1)
        for name, feat in feature_specs.items()
    }
    tensors = {
        name: tf.convert_to_tensor(tensors[name], dtype=feat.dtype)
        for name, feat in feature_specs.items()
    }

    labels = {name: tensors.pop(name) for name in input_label_cols}

    return Dataset(tensors, labels)


def _parquet_to_tf_dataset(
    path,
    time_series_length,
    input_label_cols,
    data_split_col,
    data_shuffle_col,
    exclude_columns,
    include_columns=None,
    seed=None,
):
    pf = pq.ParquetFile(pyarrow.py_buffer(tf.io.read_file(path).numpy()))
    arrow_schema = pf.schema.to_arrow_schema()
    all_features = _arrow_schema_to_tf(arrow_schema, time_series_length)
    for feat in exclude_columns:
        all_features.pop(feat)
    if include_columns is None:
        feature_specs = all_features
    else:
        feature_specs = {
            name: all_features[name]
            for name in include_columns
            + input_label_cols
            + [data_split_col, data_shuffle_col]
        }

    # read dataframe
    df = pf.read(columns=list(feature_specs.keys())).to_pandas()

    # shuffle dataframe
    data_shuffle = df.pop(data_shuffle_col)
    feature_specs.pop(data_shuffle_col)
    df = df.iloc[data_shuffle.argsort()]

    # Convert date/timestamp columns to strings
    # TODO: break out time components for input_label_cols
    for name in feature_specs.keys():
        if arrow_schema.field(name).type == pyarrow.date32():
            df[name] = pd.to_datetime(df[name]).dt.strftime('%Y-%m-%d')

    # Tensorflow doesn't support mixed value types (e.g. caused by None
    # strings), so we replace NULLs in categorial string features with an
    # out-of-vocabulary value already here, even though it would be nicer to do
    # that at the point of embedding.
    for name, feat in feature_specs.items():
        if feat.dtype == tf.string:
            df[name].fillna('__unknown', inplace=True)

    # split datasets
    data_split = df.pop(data_split_col)
    feature_specs.pop(data_split_col)
    datasets = {
        split: _pandas_to_tf_dataset(
            df[data_split == split.value], feature_specs, input_label_cols,
        )
        for split in Split
        if not df[data_split == split.value].empty
    }
    return datasets, feature_specs
