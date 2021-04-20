import typing
import os
from datetime import date, timedelta

import kerastuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as tfk
from fbprophet import Prophet
from fbprophet.make_holidays import make_holidays_df

from . import metrics
from .dataset_utils import Dataset


class Model:
    """A sales model based on `Prophet <https://facebook.github.io/prophet/>`_.

    Prophet is used for an aggregated sales forecast. This is broken
    down to per-product level using ratios of long-term7800 and short-term
    average sales.
    """

    def __init__(
        self,
        *,
        vocabularies: typing.Dict[
            str, typing.Dict[str, typing.List[typing.Union[str, float]]]
        ],
        features: typing.Dict[str, tf.io.FixedLenFeature],
        hp: kt.HyperParameters = kt.HyperParameters(),
    ):

        self.label_col = 'sales_before_returns'
        self.models: typing.Dict[tuple, Prophet] = {}

    @classmethod
    def from_models(cls, models: typing.Dict[tuple, Prophet]):
        res = cls.__new__(cls)
        res.label_col = 'sales_before_returns'
        res.models = models
        return res

    def fit(
        self, train_ds: Dataset, eval_ds: Dataset, hp: kt.HyperParameters, log_dir: str,
    ):

        self._fit(
            self._dataset_to_dataframe(train_ds),
            self._dataset_to_dataframe(eval_ds),
            hp,
            log_dir,
        )

    def _fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        hp: kt.HyperParameters,
        log_dir: str,
    ):

        daily = train_df.append(eval_df)

        for grp, df_grp in daily.groupby(['market', 'channel']):

            regressors = [
                *set(df_grp.columns) - {'market', 'channel', 'date', self.label_col}
            ]
            prophet = self.models[grp] = self._build_model(
                market=grp[0], regressors=regressors, hp=hp, dates=df_grp.date,
            )

            with _suppress_stdout():
                prophet.fit(df_grp.rename(columns={'date': 'ds', self.label_col: 'y'}))

    def predict(self, ds) -> pd.DataFrame:
        return self._predict(self._dataset_to_dataframe(ds))

    def metrics(self) -> metrics.MetricsBundle:
        return SalesMetrics()

    def _prophet_predictions(self, products_daily: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame()

        for grp, df_grp in products_daily.groupby(['market', 'channel']):
            try:
                prophet = self.models[grp]
            except KeyError:
                raise ValueError(
                    f"Unknown market and channel {grp} (not present in training data)."
                )

            dates = df_grp.date.values
            df = pd.DataFrame(dict(ds=dates))
            regressors = [
                *set(df_grp.columns) - {'date', self.label_col, 'market', 'channel'}
            ]
            for regressor in regressors:
                df[regressor] = df_grp[regressor].values

            pred_df = prophet.predict(df)
            pred_df['market'] = grp[0]
            pred_df['channel'] = grp[1]
            predictions = predictions.append(pred_df)

        return predictions

    def _predict(self, products_daily: pd.DataFrame):
        predictions = self._prophet_predictions(products_daily)

        predictions = predictions[['market', 'channel', 'ds', 'yhat']]

        predictions.columns = [
            'market',
            'channel',
            'date',
            'predicted_total_sales_before_returns',
        ]

        products_daily = products_daily.set_index(['market', 'channel', 'date'])

        predictions.set_index(['market', 'channel', 'date'], inplace=True)
        predictions.predicted_total_sales_before_returns.clip(lower=0, inplace=True)
        # force nonnegatice sales predictions: https://github.com/facebook/prophet/issues/1668

        products_daily[
            'predicted_sales_before_returns'
        ] = predictions.predicted_total_sales_before_returns
        return products_daily.predicted_sales_before_returns.reset_index()

    def _build_model(
        self,
        market: str,
        regressors: list,
        hp: kt.HyperParameters,
        dates: pd.DatetimeIndex,
    ) -> Prophet:
        changepoint_prior_scale = hp.Float(
            'changepoint_prior_scale', 0.001, 0.5, sampling='log', default=0.05,
        )
        seasonality_prior_scale = hp.Float(
            'seasonality_prior_scale', 0.01, 10, sampling='log', default=10,
        )
        holidays_prior_scale = hp.Float(
            'holidays_prior_scale', 0.01, 10, sampling='log', default=10,
        )
        seasonality_mode = hp.Choice(
            'seasonality_mode', ['multiplicative', 'additive'], default='multiplicative'
        )
        daily_seasonality = hp.Boolean('daily_seasonality', default=False)
        min_days_after_last_trend_change = hp.Int(
            'min_days_after_last_trend_change', 14, 180, default=90,
        )
        regressors_mode = hp.Choice(
            'regressors_mode', ['multiplicative', 'additive'], default='multiplicative'
        )
        # add one year of future to get holidays for prediction
        holiday_years = range(dates.min().year, dates.max().year + 1 + 1)
        m = Prophet(
            changepoints=dates[
                dates <= dates.max() - timedelta(days=min_days_after_last_trend_change)
            ],
            daily_seasonality=daily_seasonality,
            yearly_seasonality=True,
            holidays=self._holidays(hp, holiday_years),
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            uncertainty_samples=0,
            weekly_seasonality=False,
        )
        # https://en.wikipedia.org/wiki/ISO_3166-1
        # workaround messy implementation of country names
        country_code_translations = dict(FR='FRA')
        country_code = country_code_translations.get(market, market)
        for regressor in regressors:
            m.add_regressor(regressor, mode=regressors_mode)
        m.add_country_holidays(country_name=country_code)
        if hp.Boolean('weekly_adjusted_seasonality', default=False):
            m.add_seasonality(
                name='weekly', period=7, fourier_order=7, prior_scale=1.2
            )  # adjusted values
        else:
            m.add_seasonality(
                name='weekly', period=7, fourier_order=3
            )  # default values
        if hp.Boolean('4monthly_seasonality', default=False):
            m.add_seasonality(
                name='4monthly_seasonality', period=120, fourier_order=5, prior_scale=2
            )  # adjusted values
        return m

    def _holidays(
        self, hp: kt.HyperParameters, years: typing.Iterable[int]
    ) -> pd.DataFrame:
        holidays = pd.DataFrame()
        if hp.Boolean('holidays_black_fridays', default=True):
            holidays = holidays.append(
                self._get_black_fridays(
                    years, hp.Boolean('holidays_black_friday_periods', default=True)
                ),
            )
        if hp.Boolean('holidays_christmas_days', default=True):
            holidays = holidays.append(self._get_christmas_days(years))
        if hp.Boolean('holidays_boxing_days', default=False):
            holidays = holidays.append(self._get_boxing_days(years))
        return None if holidays.empty else holidays

    @staticmethod
    def _get_black_fridays(
        years: typing.Iterable[int], extend_period: typing.Optional[bool] = False,
    ):
        df_us_holidays = make_holidays_df(years, 'US')
        return pd.DataFrame(
            dict(
                holiday='Black Friday',
                ds=df_us_holidays[df_us_holidays.holiday == 'Thanksgiving'].ds.dt.date
                + timedelta(days=1),
                lower_window=-4 if extend_period else 0,
                upper_window=3 if extend_period else 0,
            )
        )

    @staticmethod
    def _get_christmas_days(years: typing.Iterable[int]):
        return pd.DataFrame(
            dict(
                holiday='Christmas',
                ds=pd.to_datetime([date(year, 12, 24) for year in years]),
                lower_window=-3,  # often effect in days before christmas
                upper_window=0,  # covered by public holidays
            )
        )

    @staticmethod
    def _get_boxing_days(years: typing.Iterable[int]):
        return pd.DataFrame(
            dict(
                holiday='Boxing Days',
                ds=pd.to_datetime([date(year, 12, 27) for year in years]),
                lower_window=0,
                upper_window=3,
            )
        )

    def _dataset_to_dataframe(self, ds: Dataset) -> pd.DataFrame:
        X, Y = ds[:]
        if not isinstance(Y, dict):
            Y = {self.label_col: Y}
        df = pd.DataFrame(
            {
                name: tf.squeeze(tensor, -1).numpy()
                for name, tensor in {**X, **Y}.items()
            }
        )
        for col in df.select_dtypes(object).columns:
            df[col] = df[col].str.decode('utf-8')
        df.date = pd.to_datetime(df.date)
        # workaround float32 not being JSON searializable
        # (https://ellisvalentiner.github.io/post/2016-01-20-numpyfloat64-is-json-serializable-but-numpyfloat32-is-not/)
        for col in df.select_dtypes(np.float32).columns:
            df[col] = df[col].astype('float64')
        return df


class SalesMetrics(metrics.MetricsBundle):
    def __init__(self):
        super(SalesMetrics, self).__init__()
        self.loss = tfk.metrics.Mean('loss', dtype=tf.float32)
        self.mae, self.z_mae, self.nz_mae = metrics.split_zero_mask(
            tfk.metrics.MeanAbsoluteError, 'mae'
        )
        self.mse, self.z_mse, self.nz_mse = metrics.split_zero_mask(
            tfk.metrics.MeanSquaredError, 'mse'
        )
        self.max_mse = metrics.Max('max_mse')
        self.agg_diff = metrics.AggregatedPercentageError('agg_diff')
        self.smape = tfk.metrics.Mean('smape')

    def update_state(self, y_true, y_pred, loss):
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)
        self.loss.update_state(loss)
        self.mae.update_state(y_true, y_pred)
        self.nz_mae.update_state(y_true, y_pred)
        self.mse.update_state(y_true, y_pred)
        self.nz_mse.update_state(y_true, y_pred)
        self.max_mse.update_state(tf.square(y_true - y_pred))
        self.agg_diff.update_state(y_true, y_pred)
        self.smape.update_state(metrics.smape(y_true, y_pred))

    def all_metrics(self):
        return [
            self.loss,
            self.mae,
            self.z_mae,
            self.nz_mae,
            self.mse,
            self.z_mse,
            self.nz_mse,
            self.max_mse,
            self.agg_diff,
            self.smape,
        ]


# https://github.com/facebook/prophet/issues/223#issuecomment-326455744
class _suppress_stdout(object):
    '''
    A context manager for doing a "deep suppression" of stdout in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
    '''

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(1)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        # os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        # os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
