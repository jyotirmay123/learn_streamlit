# Copyright © 2019 7Learnings GmbH
import os
import shutil
import tempfile
import unittest

import kerastuner as kt
import tensorflow as tf

from .datasets import Split, synthetic_price_periods
from models.price_period_elasticity import Model
from models.train_loop import train
from models.utils import save_model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.plot_dir = 'tmp/plots'
        self.metrics_dir = 'tmp/metrics'
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_predict(self):
        model, _ = self._test_train(
            synthetic_price_periods, epochs=0, eager=True, max_num_batches=1,
        )
        eval_ds, *_ = synthetic_price_periods(split=Split.eval, batch_size=16)

        for i in range(0, 32, 16):
            X, Y = eval_ds[i : i + 16]
            model.predict(**X)

    def test_save(self):
        model, _ = self._test_train(
            synthetic_price_periods, epochs=1, eager=True, max_num_batches=1,
        )
        path = os.path.join(self.test_dir, 'test_save')
        save_model(model, path)
        assert tf.saved_model.contains_saved_model(os.path.join(path, 'model'))

    def _test_train(
        self, dataset_function, epochs, eager, max_num_batches=None, log_dir=None
    ):
        """helper function to reduce test boilerplate"""
        train_ds, features, vocabularies = dataset_function(
            split=Split.train, num_price_changes=10000,
        )
        eval_ds, *_ = dataset_function(split=Split.eval, num_price_changes=1000)
        batch_size = 128
        if max_num_batches is not None:
            train_ds.length = max_num_batches * batch_size
            eval_ds.length = max_num_batches * batch_size

        model = Model(vocabularies=vocabularies, features=features,)

        hp = kt.HyperParameters()
        hp.Fixed('batch_size', batch_size)
        hp.Fixed('num_epochs', epochs)
        hp.Fixed('learning_rate', 5e-3)
        hp.Fixed('optimizer', 'Adam')
        history = train(model, train_ds, eval_ds, hp, eager=eager, log_dir=log_dir)
        return model, history
