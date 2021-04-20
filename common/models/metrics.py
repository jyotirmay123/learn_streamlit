# Copyright © 2019 7Learnings GmbH
import abc
import logging
import math
import typing

import hypertune
import tensorflow as tf
import tensorflow.keras as tfk


class MetricsBundle:
    """
    Bundled tfk.metrics objects

    Used to relate specific metrics (e.g. mse, mae, and mape) to each model.
    """

    def __init__(self):
        self.history = History()
        self.hpt = hypertune.HyperTune()

    @abc.abstractmethod
    def update_state(self, y_actual, y_predicted, loss, **kwargs):
        raise NotImplementedError("Please Implement this method")

    @abc.abstractmethod
    def all_metrics(self):
        raise NotImplementedError("Please Implement this method")

    def output_metrics(
        self,
        epoch: int,
        epochs: int,
        evaluation: bool,
        time: float = 0,
        summary_writer: tf.summary.SummaryWriter = None,
        logger: logging.Logger = None,
    ):
        if summary_writer is not None:
            self._write_tf_summary(summary_writer, epoch, evaluation=False)
        if logger is not None:
            self._log_metrics(logger, epoch, epochs, time, evaluation)
        self._record_history(epoch, evaluation)
        for v in self.all_metrics():
            v.reset_states()

    def _write_tf_summary(self, summary_writer, epoch, evaluation):
        with summary_writer.as_default():
            for m in self.all_metrics():
                tf.summary.scalar(m.name, m.result(), step=epoch)
                if evaluation:
                    self.hpt.report_hyperparameter_tuning_metric(
                        hyperparameter_metric_tag=m.name,
                        metric_value=m.result(),
                        global_step=epoch,
                    )
        summary_writer.flush()

    def _log_metrics(self, logger, epoch, epochs, time, evaluation):
        if evaluation:
            line = f"Eval          {time:6.2f} s: "
        else:
            line = f"Train {epoch:3}/{epochs:3} {time:6.2f} s: "
        line += ', '.join(
            f"{m.name} {m.result().numpy():6.3f}" for m in self.all_metrics()
        )
        logger.info(line)

    def _record_history(self, epoch, evaluation):
        self.history.update(
            epoch=epoch,
            evaluation=evaluation,
            metrics={m.name: m.result().numpy() for m in self.all_metrics()},
        )


class History:
    def __init__(self):
        self.epochs = []
        self.train = {}
        self.eval = {}
        self.best_epoch = 0

    def update(self, epoch: int, evaluation: bool, metrics: typing.Dict[str, float]):
        if not evaluation:
            self.epochs.append(epoch)
        hist = self.eval if evaluation else self.train
        for name, value in metrics.items():
            hist.setdefault(name, []).append(value)

    def get_final_eval(self) -> typing.Optional[typing.Dict[str, float]]:
        if len(self.epochs) == 0 or self.best_epoch < self.epochs[0]:
            return None
        idx = self.epochs.index(self.best_epoch)
        return {name: vals[idx] for name, vals in self.eval.items()}


def mase(y_true, y_pred, y_naive):
    """https://en.wikipedia.org/wiki/Mean_absolute_scaled_error"""
    assert len(y_true.shape) == 2, 'Expected 2 dimensional tensors'
    # MAE of prediction
    err_pred = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
    # MAE of naive forecast
    err_naive = tf.reduce_mean(tf.abs(y_true - y_naive), axis=-1)
    # clip minimal naive error to 1 to account for high mispredictions with constant time-series
    return err_pred / tf.clip_by_value(err_naive, 1, float('inf'))


def smape(y_true, y_pred):
    """https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error"""
    return tf.reduce_mean(
        2
        * tf.abs(y_pred - y_true)
        / tf.clip_by_value(tf.abs(y_pred) + tf.abs(y_true), 1, float('inf')),
        axis=-1,
    )


def _approx_log_gamma(n):
    """
    Computes approximate log(𝚪(n)) using Stirling's approximation.
    Also see: https://en.wikipedia.org/wiki/Stirling%27s_approximation
    """
    return n * tf.math.log(n) - n + 0.5 * tf.math.log(2 * math.pi * n)


def elasticity_loss(X, y_pred, compute_full_loss=False):
    """
    Negative log likelihood of predicted elasticities for observed sales around price changes.
    Compute absolute rather than proportional loss when compute_full_loss is true.
    Also see: leipzig-01/streamlit/price_period_elasticity.py for derivation loss formula
    """
    # avoid nan results below with tanh = ±1 due to float/tanh inaccuracies (fixes #112)
    pct_changes = tf.math.tanh(
        tf.cast(X['price_change'], tf.float32) * tf.cast(y_pred, tf.float32) / 2
    ) * (1 - 3 * tfk.backend.epsilon())
    alphas = (
        (1 + pct_changes)
        / (1 - pct_changes)
        * tf.cast(X['num_days'], tf.float32)
        / tf.cast(X['prev_num_days'], tf.float32)
    )
    k1 = tf.cast(X['prev_sales_before_returns'], tf.float32)
    k2 = tf.cast(X['sales_before_returns'], tf.float32)
    # log(𝚪(k1 + k2 + 1) * α^k_2 / α^(k_1 + k_2 + 1) / k_1! / k_2!)
    #
    # log(α^k_2 / α^(k_1 + k_2 + 1))
    logprobs = tf.math.log(alphas) * k2 - tf.math.log(1 + alphas) * (k1 + k2 + 1)
    if compute_full_loss:
        # log(𝚪(k1 + k2 + 1) / k1! / k2!)
        logprobs += (
            _approx_log_gamma(k1 + k2 + 1)
            - _approx_log_gamma(k1 + 1)
            - _approx_log_gamma(k2 + 1)
        )
    return -logprobs


class AggregatedPercentageError(tfk.metrics.Metric):
    """
    Metric to track relative deviation of overall/summed predicted and actual values.
    """

    def __init__(self, name='aggregated_percentage_error', **kwargs):
        super(AggregatedPercentageError, self).__init__(name=name, **kwargs)
        self.true_sum = tf.Variable(0.0)
        self.pred_sum = tf.Variable(0.0)

    def update_state(self, y_true, y_pred):
        """
        Update metric for one micro-batch, adds y_true (actual) and
        y_pred (predicted) values to internal sums, to later compute
        the ratio as relative deviation.
        """
        self.true_sum.assign_add(tf.reduce_sum(y_true))
        self.pred_sum.assign_add(tf.reduce_sum(y_pred))

    def result(self):
        return self.pred_sum / self.true_sum


class Min(tfk.metrics.Metric):
    """
    Computes the min of the given values.
    """

    def __init__(self, name='min', **kwargs):
        super(Min, self).__init__(name=name, **kwargs)
        self.min = tf.Variable(float('inf'))

    def update_state(self, values: tf.Tensor):
        """Update metric state with new values."""
        self.min.assign(tf.math.minimum(self.min, tf.math.reduce_min(values)))

    def result(self):
        return self.min


class Max(tfk.metrics.Metric):
    """
    Computes the max of the given values.
    """

    def __init__(self, name='max', **kwargs):
        super(Max, self).__init__(name=name, **kwargs)
        self.max = tf.Variable(float('-inf'))

    def update_state(self, values: tf.Tensor):
        """Update metric state with new values."""
        self.max.assign(tf.math.maximum(self.max, tf.math.reduce_max(values)))

    def result(self):
        return self.max


class RSquared(tfk.metrics.Metric):
    """
    Computes the R2-score between `y_true` and `y_pred`

    See https://en.wikipedia.org/wiki/Coefficient_of_determination
    """

    def __init__(self, name='r2', **kwargs):
        super(RSquared, self).__init__(name=name, **kwargs)
        self.count = tf.Variable(0)
        self.sum = tf.Variable(0.0, dtype=tf.float64)
        self.sum_sq = tf.Variable(0.0, dtype=tf.float64)
        self.sum_res_sq = tf.Variable(0.0, dtype=tf.float64)

    def update_state(self, y_true, y_pred):
        """
        Update metric for one micro-batch, adds residuals and
        total sum of squares to internal sums.
        """
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.cast(y_pred, tf.float64)
        self.count.assign_add(tf.shape(y_true)[0])
        self.sum.assign_add(tf.reduce_sum(y_true))
        self.sum_sq.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.sum_res_sq.assign_add(tf.reduce_sum(tf.square(y_pred - y_true)))

    def result(self):
        ss_total = self.sum_sq - self.sum * (self.sum / tf.cast(self.count, tf.float64))
        ss_res = self.sum_res_sq
        return 1 - (ss_res / ss_total)


class MaskedMetric:
    def __init__(
        self,
        get_mask: typing.Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        metric: tfk.metrics.Metric,
    ):
        self.metric = metric
        self.get_mask = get_mask

    @property
    def name(self):
        return self.metric.name

    def update_state(self, y_true, y_pred):
        mask = self.get_mask(y_true, y_pred)
        if tf.reduce_any(mask):
            self.metric.update_state(
                tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask)
            )

    def reset_states(self):
        self.metric.reset_states()

    def result(self):
        return self.metric.result()


def zero_label(y_true, y_pred):
    return y_true == 0


def nonzero_label(y_true, y_pred):
    return y_true != 0


def split_zero_mask(
    metric_class, name
) -> typing.Tuple[MaskedMetric, MaskedMetric, MaskedMetric]:
    """Construct 3 metrics for unmasket, zero-, and non-zero masked
    versions of the given `metric_class`"""
    return (
        metric_class(name),
        MaskedMetric(zero_label, metric_class(f"z_{name}")),
        MaskedMetric(nonzero_label, metric_class(f"nz_{name}")),
    )
