#!/usr/bin/env python3

import json
import argparse
from google.cloud import bigquery as bq, bigquery_storage as bqs


class DataFrameIterator(list):
    def __init__(self, it):
        super().__init__()
        self.it = it

    def __iter__(self):
        for i, d in enumerate(self.it):
            rows = d.to_dict(orient='records')
            for r in rows:
                yield r

    def __len__(self):
        return 1


class SchemaIterator(list):
    def __init__(self, schema):
        super().__init__()
        self.schema = schema

    def __iter__(self):
        for s in self.schema:
            yield s.to_api_repr()

    def __len__(self):
        return 1


def save_schema_json(output, schema):
    filename = "{}_schema.json".format(output)
    with open(filename, mode='w') as fp:
        json.dump(SchemaIterator(schema), fp, indent=2)

    return filename


def save_data_json(output, df_iter):
    filename = "{}_data.json".format(output)
    with open(filename, mode='w') as fp:
        json.dump(DataFrameIterator(df_iter), fp, indent=2)

    return filename


def save_data_ndjson(output, df_iter):
    filename = "{}_data.ndjson".format(output)
    with open(filename, mode='w') as fp:
        for df in df_iter:
            rows = df.to_dict(orient='records')
            for r in rows:
                json.dump(r, fp)
                fp.write('\n')

    return filename


def parse():
    parser = argparse.ArgumentParser(
        description='Dump BigQuery result among with its schema'
    )
    parser.add_argument('output', type=str, help='Prefix of output filename')
    parser.add_argument('query', type=str, help='query statement of BigQuery')
    parser.add_argument(
        '--ndjson',
        action='store_true',
        help='Save dumped data to ndJSON (Newline Delimited JSON) format',
    )
    parser.add_argument(
        '--limit', type=int, default=10, help='Limit the number of rows (default: 10)'
    )
    return parser.parse_args()


def main():
    args = parse()

    query = args.query
    if args.limit > 0:
        query = "{} LIMIT {}".format(args.query, args.limit)

    client = bq.Client()
    result = client.query(query, job_config=bq.QueryJobConfig()).result(timeout=1800)

    # Use streaming result to avoid memory overhead
    if hasattr(result, 'to_dataframe_iterable'):
        df_iter_func = result.to_dataframe_iterable
    else:
        df_iter_func = result._to_dataframe_iterable

    df_iter = df_iter_func(bqstorage_client=bqs.BigQueryStorageClient(), dtypes={})

    if args.ndjson:
        save_data_func = save_data_ndjson
    else:
        save_data_func = save_data_json

    schema_filename = save_schema_json(args.output, result.schema)
    data_filename = save_data_func(args.output, df_iter)
    print(
        "Dump completed, schema_file={}, data_file={}".format(
            schema_filename, data_filename
        )
    )


if __name__ == "__main__":
    main()
