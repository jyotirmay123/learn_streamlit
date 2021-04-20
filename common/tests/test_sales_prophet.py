# Copyright © 2020 7Learnings GmbH
import os
import shutil
import tempfile
import unittest

import kerastuner as kt
import numpy as np
from fbprophet.serialize import model_to_json

from .datasets import Split, synthetic_simple_total_sales
from models.sales_prophet import Model
from models.train_loop import train
from models.utils import save_model, load_model_prophet


class TestModel(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.plot_dir = 'tmp/plots'
        self.metrics_dir = 'tmp/metrics'
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_train_history(self):
        _, history = self._test_train(synthetic_simple_total_sales, max_num_batches=1)
        assert history.epochs == [1]
        assert len(history.train['loss']) == 1
        assert isinstance(history.train['loss'][0], (np.floating, float))
        assert isinstance(history.train['loss'][-1], (np.floating, float))
        for m in ['loss', 'mse', 'mae']:
            assert len(history.train[m]) == 1
            assert len(history.eval[m]) == 1

    def test_predict(self):
        model, _ = self._test_train(synthetic_simple_total_sales)
        eval_ds, *_ = synthetic_simple_total_sales(split=Split.eval, batch_size=16)
        model.predict(eval_ds)

    def test_save(self):
        model, _ = self._test_train(synthetic_simple_total_sales, max_num_batches=1)
        path = os.path.join(self.test_dir, 'test_save')
        save_model(model, path)
        assert os.path.exists(f"{path}/model.json")
        model2 = load_model_prophet(path)
        assert {grp: model_to_json(m) for grp, m in model.models.items()} == {
            grp: model_to_json(m) for grp, m in model2.models.items()
        }

    @staticmethod
    def _test_train(dataset_function, max_num_batches=None, log_dir=None):
        """helper function to reduce test boilerplate"""
        train_ds, features, vocabularies = dataset_function(split=Split.train)
        eval_ds, *_ = dataset_function(split=Split.eval)
        batch_size = 128
        if max_num_batches is not None:
            train_ds.length = max_num_batches * batch_size
            eval_ds.length = max_num_batches * batch_size

        # TODO: add first or last time-step of norm_dayofyear and
        # norm_year to dataset so they can be used as context features
        model = Model(vocabularies=vocabularies, features=features)

        hp = kt.HyperParameters()
        hp.Fixed('batch_size', batch_size)
        history = train(model, train_ds, eval_ds, hp, eager=False, log_dir=log_dir)
        return model, history
