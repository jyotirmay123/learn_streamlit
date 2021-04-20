# Copyright © 2019 7Learnings GmbH
import typing

import kerastuner as kt
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from . import metrics, utils

TEST_PERCENTILES = [10, 25, 50, 75, 90]


class Model(tf.Module):
    '''Model to predict elasticity trained on neg-log-likelihood of
       observed sales_before_returns around a price change.'''

    def __init__(
        self,
        vocabularies: typing.Dict[str, typing.Dict[str, typing.List[typing.Any]]],
        features: typing.Dict[str, tf.io.FixedLenFeature],
        reference_elasticity=-2.0,
        hp: kt.HyperParameters = kt.HyperParameters(),
    ):
        super(Model, self).__init__()
        loss_features: typing.Dict[str, typing.List[typing.Optional[int]]] = dict(
            prev_num_days=[1],
            num_days=[1],
            prev_sales_before_returns=[1],
            sales_before_returns=[1],
            price_change=[1],
            elasticity=[1],
        )
        utils.check_expected_feature_shapes(features, loss_features)

        self.features = {
            name: feat for name, feat in features.items() if name not in loss_features
        }
        self.reference_elasticity = reference_elasticity

        self.vocabularies = vocabularies
        self.lookup_tables, self.embeddings = utils.create_embedding_layers(
            self.features, {name: voc['labels'] for name, voc in vocabularies.items()},
        )

        num_layers = hp.Int('num_layers', 0, 10, default=3)
        num_units = hp.Int('num_units', 1, 512, sampling='log', default=64)
        activation = hp.Choice('activation', ['tanh', 'linear', 'relu'], default='relu')
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
                activation='linear',
                kernel_regularizer=tfk.regularizers.l2(l2_reg),
                kernel_initializer='random_normal',
                bias_initializer='zeros',
            )
        ]

    def __call__(self, X, training=True):
        assert 'elasticity' not in X, 'Unexpected elasticity feature'
        elasticity = utils.concat_context_with_time_series_features(
            X, self.features, self.lookup_tables, self.embeddings, training=training,
        )
        for layer in self.layers:
            elasticity = layer(elasticity, training=training)

        return tf.squeeze(elasticity, axis=1)

    def loss(self, Y_true, Y_pred):
        return metrics.elasticity_loss(Y_true, Y_pred) - metrics.elasticity_loss(
            Y_true, self.reference_elasticity
        )

    @tf.function
    def predict(self, **X):
        elasticity = self(X, training=False)
        return {'predicted_elasticity': elasticity}

    def metrics(self):
        return Metrics()

    early_stopping_config = {'loss': min, 'max_loss': min}


class Metrics(metrics.MetricsBundle):
    def __init__(self):
        super(Metrics, self).__init__()

        self.loss = tfk.metrics.Mean('loss', dtype=tf.float32)
        self.min_loss = metrics.Min('min_loss', dtype=tf.float32)
        self.max_loss = metrics.Max('max_loss', dtype=tf.float32)
        self.avg_pred_ela = tfk.metrics.Mean('avg_pred_ela', dtype=tf.float32)
        self.min_pred_ela = metrics.Min('min_pred_ela', dtype=tf.float32)
        self.max_pred_ela = metrics.Max('max_pred_ela', dtype=tf.float32)
        self.mae = tfk.metrics.MeanAbsoluteError('mae')
        self.mse = tfk.metrics.MeanSquaredError('mse')

    def update_state(self, y_true, y_pred, total_loss):
        self.loss.update_state(tf.reduce_mean(total_loss))
        self.min_loss.update_state(total_loss)
        self.max_loss.update_state(total_loss)
        self.avg_pred_ela.update_state(y_pred)
        self.min_pred_ela.update_state(y_pred)
        self.max_pred_ela.update_state(y_pred)
        self.mae.update_state(y_true['elasticity'], y_pred)
        self.mse.update_state(y_true['elasticity'], y_pred)

    def all_metrics(self):
        return [
            self.loss,
            self.min_loss,
            self.max_loss,
            self.avg_pred_ela,
            self.min_pred_ela,
            self.max_pred_ela,
            self.mae,
            self.mse,
        ]
