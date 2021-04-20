# Copyright © 2019 7Learnings GmbH
"""Datasets for model testing"""
from datetime import date, timedelta
import itertools
import typing

import numpy as np
import scipy.stats
import tensorflow as tf

from models.dataset_utils import Split, Dataset


def synthetic_sales(  # noqa: C901
    split: Split,
    batch_size: int = 1,
    num_products: int = 1000,
    context_length: int = 21,
    prediction_length: int = 7,
):
    vocabularies = [
        ('color', ['red', 'blue', 'green', 'yellow', 'orange', 'black', 'white']),
        ('brand', ['brand_' + chr(ord('A') + i) for i in range(26)]),
        ('category', ['category_' + chr(ord('A') + i) for i in range(10)]),
    ]
    rnd = np.random.RandomState(42)
    data = {
        name: rnd.choice(vocabulary, size=(num_products, 1))
        for name, vocabulary in vocabularies
    }
    data['product_id'] = np.reshape(np.arange(num_products), (-1, 1))
    data['date'] = np.tile(
        np.datetime_as_string(
            np.arange(np.datetime64('2018-01-01'), np.datetime64('2019-01-01'))
        ),
        (num_products, 1),
    )
    # TODO: non-stationary sales rate (e.g. periodicity, trend)
    data['base_sales_rate'] = rnd.pareto(a=1.5, size=(num_products, 1)).astype(
        np.float32
    )
    # TODO: more random discounts
    data['norm_discount'] = np.concatenate(
        (
            np.zeros((num_products, data['date'].shape[1] // 2, 1), dtype=np.float32),
            np.ones(
                (num_products, (data['date'].shape[1] + 1) // 2, 1), dtype=np.float32
            )
            * 0.1,
        ),
        axis=1,
    )
    data['norm_availability'] = rnd.beta(
        a=10, b=1, size=(*data['date'].shape, 1)
    ).astype(np.float32)
    data['seasonality'] = np.ones(
        (*data['date'].shape, 1), dtype=np.float32
    ) * 0.5 + np.sin(np.linspace(0.0, np.pi, data['date'].shape[1])).reshape(
        (1, -1, 1)
    ).astype(
        np.float32
    )
    data['sales_rate'] = (
        data['base_sales_rate'].reshape((-1, 1, 1)) * data['seasonality']
    )
    data['sales_before_returns'] = rnd.poisson(data['sales_rate']).astype(np.float32)
    # https://stackoverflow.com/a/37977222/2371032
    num_non_zero_sales = (data['sales_before_returns'] > 0).sum(
        axis=1, dtype=np.float32
    )
    sales_before_returns_scaling = np.divide(
        data['sales_before_returns'].sum(axis=1, dtype=np.float32),
        num_non_zero_sales,
        out=np.zeros_like(num_non_zero_sales),
        where=num_non_zero_sales > 0,
    ).reshape((-1, 1, 1))
    data['norm_historical_sales_before_returns'] = np.divide(
        data['sales_before_returns'],
        sales_before_returns_scaling,
        out=np.zeros_like(data['sales_before_returns']),
        where=sales_before_returns_scaling > 0,
    )
    data['avg_sales_before_returns'] = data['sales_before_returns'].mean(axis=1)

    # windowing
    if split == Split.train:
        num_windows = (
            data['sales_before_returns'].shape[1]
            - context_length
            - 3 * prediction_length
        )
        for name, ary in data.items():
            if len(ary.shape) >= 2 and ary.shape[1] > 1:
                data[name] = np.stack(
                    [
                        ary[:, i : i + context_length + prediction_length]
                        for i in range(num_windows)
                    ],
                    axis=1,
                ).reshape(
                    (num_windows * num_products, context_length + prediction_length)
                    + ary.shape[2:]
                )
            else:
                data[name] = np.repeat(ary, num_windows, axis=0)
        samples = np.argsort(rnd.rand(num_products * num_windows))[:10000]
        for name, ary in data.items():
            data[name] = ary[samples]
    elif split == Split.eval:
        for name, ary in data.items():
            if len(ary.shape) >= 2 and ary.shape[1] > 1:
                data[name] = ary[
                    :, -context_length - 2 * prediction_length : -prediction_length
                ]
    else:
        raise NotImplementedError(split.name)
    data = {name: tf.convert_to_tensor(t) for name, t in data.items()}
    features = data
    labels = features.pop('sales_before_returns')
    ds = Dataset(features, labels)
    feature_specs = _synthetic_sales_features()
    vocabularies = {name: {'labels': vocabulary} for name, vocabulary in vocabularies}
    return ds, feature_specs, vocabularies


def _synthetic_sales_features(time_series_length=28):
    time_series_features = [
        'sales_before_returns',
        'norm_historical_sales_before_returns',
        'norm_discount',
        'norm_availability',
        'seasonality',
    ]
    context_features = [
        'avg_sales_before_returns',
    ]
    categorial_features = {
        'product_id': tf.int64,
        'color': tf.string,
        'brand': tf.string,
        'category': tf.string,
    }
    categorial_ts_features = {}
    features = dict(
        [('date', tf.io.FixedLenFeature([time_series_length], tf.string))]
        + [
            (name, tf.io.FixedLenFeature([time_series_length, 1], tf.float32))
            for name in time_series_features
        ]
        + [(name, tf.io.FixedLenFeature([1], tf.float32)) for name in context_features]
        + [
            (name, tf.io.FixedLenFeature([1], dtype))
            for name, dtype in categorial_features.items()
        ]
        + [
            (name, tf.io.FixedLenFeature([time_series_length, 1], dtype))
            for name, dtype in categorial_ts_features.items()
        ]
    )
    return features


def synthetic_simple_sales(
    split: Split, batch_size: int = 1, num_products: int = 1000,
):
    rnd = np.random.RandomState(42)

    if split == Split.train:
        start_date = np.datetime64('2018-01-01')
        end_date = np.datetime64(date.fromisoformat('2019-01-01') - timedelta(days=14))
    elif split == Split.eval:
        start_date = np.datetime64(
            date.fromisoformat('2019-01-01') - timedelta(days=14)
        )
        end_date = np.datetime64('2019-01-01')
    else:
        raise NotImplementedError(split.name)

    dates = np.arange(start_date, end_date)

    categories = [
        ('market', ['DE', 'FR']),
        ('channel', ['online', 'offline']),
    ]
    data = {
        name: np.repeat(rnd.choice(category, size=num_products), len(dates))
        for name, category in categories
    }

    data['date'] = np.tile(np.datetime_as_string(dates), num_products)
    data['product_id'] = np.repeat(np.arange(num_products), len(dates))
    base_sales_rate = np.repeat(
        rnd.pareto(a=1.5, size=num_products).astype(np.float32), len(dates)
    )
    # TODO: add weekly and holiday seasonality
    seasonality = np.tile(
        0.5
        + np.sin(
            (dates - np.datetime64('2018-01-01')).astype(np.float32) * np.pi / 365
        ),
        num_products,
    )
    data['sales_before_returns'] = rnd.poisson(base_sales_rate * seasonality).astype(
        np.float32
    )
    data['avg_sales_before_returns_scale'] = np.repeat(
        data['sales_before_returns'].reshape((num_products, len(dates))).mean(axis=1)
        / data['sales_before_returns'].mean(),
        len(dates),
    )

    # add singular feature dim to facilitate concatenating with embeddings
    data = {name: np.reshape(ary, (-1, 1)) for name, ary in data.items()}
    feature_specs = {
        name: tf.io.FixedLenFeature([1], tf.string if name == 'date' else tf.float32)
        for name in data.keys()
    }
    data = {name: tf.convert_to_tensor(t) for name, t in data.items()}
    features = data
    labels = features.pop('sales_before_returns')
    ds = Dataset(features, labels)
    vocabularies = {}
    return ds, feature_specs, vocabularies


def synthetic_simple_total_sales(
    split: Split, batch_size: int = 1, num_products: int = 1000,
):
    rnd = np.random.RandomState(42)

    if split == Split.train:
        start_date = np.datetime64('2018-01-01')
        end_date = np.datetime64(date.fromisoformat('2019-01-01') - timedelta(days=14))
    elif split == Split.eval:
        start_date = np.datetime64(
            date.fromisoformat('2019-01-01') - timedelta(days=14)
        )
        end_date = np.datetime64('2019-01-01')
    else:
        raise NotImplementedError(split.name)

    dates = np.arange(start_date, end_date)

    channels = list(itertools.product(['DE', 'FR'], ['online', 'offline']))

    data = {}
    data['market'] = np.empty(0, np.str)
    data['channel'] = np.empty(0, np.str)
    for market, channel in channels:
        data['market'] = np.append(data['market'], np.repeat(market, len(dates)))
        data['channel'] = np.append(data['channel'], np.repeat(channel, len(dates)))

    data['date'] = np.tile(np.datetime_as_string(dates), len(channels))
    base_sales_rate = np.repeat(
        1000 * rnd.pareto(a=1.5, size=len(channels)).astype(np.float32), len(dates)
    )
    # TODO: add weekly and holiday seasonality
    seasonality = np.tile(
        0.5
        + np.sin(
            (dates - np.datetime64('2018-01-01')).astype(np.float32) * np.pi / 365
        ),
        len(channels),
    )
    data['sales_before_returns'] = rnd.poisson(base_sales_rate * seasonality).astype(
        np.float32
    )

    # add singular feature dim to facilitate concatenating with embeddings
    data = {name: np.reshape(ary, (-1, 1)) for name, ary in data.items()}
    feature_specs = {
        name: tf.io.FixedLenFeature([1], tf.string if name == 'date' else tf.float32)
        for name in data.keys()
    }
    data = {name: tf.convert_to_tensor(t) for name, t in data.items()}
    features = data
    labels = features.pop('sales_before_returns')
    ds = Dataset(features, labels)
    vocabularies = {}
    return ds, feature_specs, vocabularies


def _synthetic_simple_sales(
    split: Split, batch_size: int = 1, num_products: int = 1000,
) -> typing.Dict[str, np.ndarray]:
    rnd = np.random.RandomState(42)

    if split == Split.train:
        start_date = np.datetime64('2018-01-01')
        end_date = np.datetime64(date.fromisoformat('2019-01-01') - timedelta(days=14))
    elif split == Split.eval:
        start_date = np.datetime64(
            date.fromisoformat('2019-01-01') - timedelta(days=14)
        )
        end_date = np.datetime64('2019-01-01')
    else:
        raise NotImplementedError(split.name)

    dates = np.arange(start_date, end_date)

    categories = [
        ('market', ['DE', 'FR']),
        ('channel', ['online', 'offline']),
    ]
    data = {
        name: np.repeat(rnd.choice(category, size=num_products), len(dates))
        for name, category in categories
    }

    data['date'] = np.tile(np.datetime_as_string(dates), num_products)
    data['product_id'] = np.repeat(np.arange(num_products), len(dates))
    base_sales_rate = np.repeat(
        rnd.pareto(a=1.5, size=num_products).astype(np.float32), len(dates)
    )
    # TODO: add weekly and holiday seasonality
    seasonality = np.tile(
        0.5
        + np.sin(
            (dates - np.datetime64('2018-01-01')).astype(np.float32) * np.pi / 365
        ),
        num_products,
    )
    data['sales_before_returns'] = rnd.poisson(base_sales_rate * seasonality).astype(
        np.float32
    )
    data['avg_sales_before_returns_scale'] = np.repeat(
        data['sales_before_returns'].reshape((num_products, len(dates))).mean(axis=1)
        / data['sales_before_returns'].mean(),
        len(dates),
    )
    return data


def synthetic_price_periods(
    split: Split, batch_size: int = 1, num_price_changes: int = 1000
):

    rnd = np.random.RandomState(42)
    # distributions fitted to drugstore dataset
    data = dict(
        prev_avg_sales_before_returns=scipy.stats.expon.rvs(
            scale=1.0, size=num_price_changes, random_state=rnd
        ),
        price_change=rnd.choice(
            np.concatenate(
                [
                    -scipy.stats.expon.rvs(
                        scale=0.12, loc=0.01, size=num_price_changes, random_state=rnd
                    ),
                    scipy.stats.expon.rvs(
                        scale=0.09,
                        loc=0.01,
                        size=2 * num_price_changes,
                        random_state=rnd,
                    ),
                ]
            ),
            size=num_price_changes,
            replace=False,
        ),
        prev_num_days=np.minimum(
            scipy.stats.randint.rvs(1, 40, size=num_price_changes, random_state=rnd,),
            28,
        ),
        num_days=np.minimum(
            scipy.stats.randint.rvs(1, 40, size=num_price_changes, random_state=rnd,),
            28,
        ),
        prev_gross_red_price=scipy.stats.betaprime.rvs(
            15.7, 2, size=num_price_changes, random_state=rnd,
        ),
        prev_avg_basket_position=scipy.stats.norm.rvs(
            loc=3.1, scale=0.8, size=num_price_changes, random_state=rnd,
        ),
        prev_min_comp_price_ratio=np.maximum(
            scipy.stats.norm.rvs(
                loc=1.22, scale=0.2, size=num_price_changes, random_state=rnd,
            ),
            0,
        ),
    )
    # modeled roughly after bucketized regressions
    data['elasticity'] = (
        -1.5
        + -0.5 * (np.log2(data['prev_gross_red_price']) - np.log2(5))
        + -4
        + 1 * data['prev_avg_basket_position']
        + (-1 + (data['prev_min_comp_price_ratio'] - 0.85) * 1)
        * (np.log2(data['prev_gross_red_price']) - np.log2(5))
    )

    # clip E * ΔP to avoid numerical unstability with huge price changes
    pct_changes = np.clip(data['price_change'] * data['elasticity'], -1.99, 1.99)

    prev_avg_sales_before_returns = data.pop('prev_avg_sales_before_returns')
    avg_sales_before_returns = (
        (1 + pct_changes / 2) / (1 - pct_changes / 2) * prev_avg_sales_before_returns
    )

    data['prev_sales_before_returns'] = scipy.stats.poisson.rvs(
        mu=prev_avg_sales_before_returns * data['prev_num_days'],
        size=num_price_changes,
        random_state=rnd,
    )
    data['sales_before_returns'] = scipy.stats.poisson.rvs(
        mu=avg_sales_before_returns * data['num_days'],
        size=num_price_changes,
        random_state=rnd,
    )

    # normalize features
    for name, ary in data.items():
        if name in [
            'prev_gross_red_price',
            'prev_avg_basket_position',
            'prev_min_comp_price_ratio',
            'prev_sales_before_returns',
            'sales_before_returns',
        ]:
            data[name] = (ary - ary.mean()) / ary.std()
    # add singular feature dim to facilitate concatenating with embeddings
    data = {
        name: np.reshape(ary, (-1, 1)).astype(np.float32) for name, ary in data.items()
    }
    feature_specs = {
        name: tf.io.FixedLenFeature([1], tf.float32) for name in data.keys()
    }

    data = {name: tf.convert_to_tensor(t) for name, t in data.items()}
    features = data
    labels = {
        name: data.pop(name)
        for name in [
            'elasticity',
            'prev_num_days',
            'num_days',
            'prev_sales_before_returns',
            'sales_before_returns',
            'price_change',
        ]
    }
    ds = Dataset(features, labels)
    # vocabularies = {name: {'labels': vocabulary} for name, vocabulary in vocabularies}
    vocabularies = {}
    return ds, feature_specs, vocabularies
