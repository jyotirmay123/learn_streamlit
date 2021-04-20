# Copyright © 2019 7Learnings GmbH
import typing

import kerastuner as kt
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from . import metrics, utils


class Model(tf.Module):
    ''' Dense Neural Network (DNN) regressor. '''

    def __init__(
        self,
        vocabularies: typing.Dict[str, typing.Dict[str, typing.List[typing.Any]]],
        features: typing.Dict[str, tf.io.FixedLenFeature],
        input_label_cols: typing.List[str],
        zero_label_metrics: bool,
        hp: kt.HyperParameters = kt.HyperParameters(),
    ):

        super(Model, self).__init__()
        assert (
            len(input_label_cols) <= 2
        ), f'Expected less than two feature, got {input_label_cols}'

        self.label_col = input_label_cols[0]
        self.weight_col = input_label_cols[1] if len(input_label_cols) == 2 else None
        utils.check_expected_feature_shapes(
            features, {col: [1] for col in input_label_cols}
        )
        self.zero_label_metrics = zero_label_metrics

        self.features = {
            name: feat
            for name, feat in features.items()
            if name not in input_label_cols
        }

        self.vocabularies = vocabularies
        self.lookup_tables, self.embeddings = utils.create_embedding_layers(
            self.features, {name: voc['labels'] for name, voc in vocabularies.items()},
        )

        num_layers = hp.Int('num_layers', 0, 10, default=3)
        num_units = hp.Int('num_units', 1, 512, sampling='log', default=32)
        activation = hp.Choice('activation', ['tanh', 'relu', 'linear'], default='relu')
        l2_reg = hp.Float('l2_reg', 0.0, 1.0, default=0.05)

        self.layers = []
        for i in range(num_layers):
            self.layers += [
                tfkl.BatchNormalization(),
                tfkl.Dense(
                    num_units,
                    activation=activation,
                    kernel_regularizer=tfk.regularizers.l2(l2_reg),
                ),
            ]

        self.layers += [
            tfkl.Dense(
                1,
                use_bias=hp.Boolean('output_use_bias', default=True),
                activation=hp.Choice(
                    'output_activation', ['relu', 'linear'], default='linear'
                ),
                kernel_regularizer=tfk.regularizers.l2(l2_reg),
            )
        ]

    def __call__(self, X, training=True):

        target = utils.concat_context_with_time_series_features(
            X, self.features, self.lookup_tables, self.embeddings, training=training,
        )

        for layer in self.layers:
            target = layer(target, training=training)

        return tf.squeeze(target, axis=1)

    def loss(self, Y_true: typing.Dict[str, tf.Tensor], Y_pred: tf.Tensor):
        sample_weights = Y_true[self.weight_col] if self.weight_col else None
        mse = tfk.losses.MeanSquaredError(reduction=tfk.losses.Reduction.NONE)
        loss = mse(Y_true[self.label_col], Y_pred, sample_weight=sample_weights)
        return loss

    @tf.function
    def predict(self, **X):
        return {f'predicted_{self.label_col}': self(X, training=False)}

    def metrics(self):
        return Metrics(self.label_col, self.zero_label_metrics)

    early_stopping_config = {'loss': min, 'mse': min, 'mae': min, 'r2': 1.0}


class Metrics(metrics.MetricsBundle):
    def __init__(self, label_col, zero_label_metrics: bool = False):
        super(Metrics, self).__init__()

        self.label_col = label_col
        self.zero_label_metrics = zero_label_metrics

        self.loss = tfk.metrics.Mean('loss', dtype=tf.float32)
        self.min_loss = metrics.Min('min_loss', dtype=tf.float32)
        self.max_loss = metrics.Max('max_loss', dtype=tf.float32)
        if self.zero_label_metrics:
            self.mae, self.z_mae, self.nz_mae = metrics.split_zero_mask(
                tfk.metrics.MeanAbsoluteError, 'mae'
            )
            self.mse, self.z_mse, self.nz_mse = metrics.split_zero_mask(
                tfk.metrics.MeanSquaredError, 'mse'
            )
        else:
            self.mae = tfk.metrics.MeanAbsoluteError('mae')
            self.mse = tfk.metrics.MeanSquaredError('mse')
        self.rsquared = metrics.RSquared('r2')

    def update_state(self, y_true, y_pred, total_loss):
        self.loss.update_state(total_loss)
        self.min_loss.update_state(total_loss)
        self.max_loss.update_state(total_loss)
        self.mae.update_state(y_true[self.label_col], y_pred)
        self.mse.update_state(y_true[self.label_col], y_pred)
        if self.zero_label_metrics:
            self.z_mae.update_state(y_true[self.label_col], y_pred)
            self.nz_mae.update_state(y_true[self.label_col], y_pred)
            self.z_mse.update_state(y_true[self.label_col], y_pred)
            self.nz_mse.update_state(y_true[self.label_col], y_pred)
        self.rsquared.update_state(y_true[self.label_col], y_pred)

    def all_metrics(self):
        return [
            self.loss,
            self.min_loss,
            self.max_loss,
            self.mae,
            self.mse,
            self.rsquared,
        ] + (
            [self.z_mae, self.nz_mae, self.z_mse, self.nz_mse]
            if self.zero_label_metrics
            else []
        )
