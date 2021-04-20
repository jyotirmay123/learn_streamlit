# Copyright © 2019 7Learnings GmbH
import functools
import itertools
import logging
import os
import time
import typing
from math import isclose

import google.protobuf.text_format
import kerastuner as kt
import tensorboard.plugins.projector
import tensorflow as tf
import tensorflow.keras as tfk

from . import sales_prophet
from . import metrics
from .utils import Model


logger = logging.getLogger(__name__)


# ==============================================================================
# Helper functions
# ------------------------------------------------------------------------------


def compare_metrics(
    best_metrics: typing.Optional[dict],
    eval_metrics: dict,
    early_stopping_config: typing.Dict[
        str, typing.Union[float, typing.Callable[[float, float], float]]
    ],
) -> typing.Tuple[bool, dict]:
    """
    Evaluate if there is a metric that has improved in current epoch and updated
    best metrics.
    """
    has_improved = False
    if best_metrics is None:
        best_metrics = eval_metrics
        has_improved = True
        return has_improved, best_metrics

    eval_loss = eval_metrics['loss']
    best_loss = best_metrics['loss']
    # check if loss has improved witin a certain range
    if eval_loss < best_loss and not isclose(eval_loss, best_loss, rel_tol=3e-03):
        has_improved = True
        best_metrics['loss'] = eval_metrics['loss']

    # check if any of the other metrics have improved
    elif isclose(eval_loss, best_loss, rel_tol=3e-03):
        for metric_name in eval_metrics.keys():

            if metric_name == 'loss':  # skip loss already checked
                continue

            optimization_rule = early_stopping_config[metric_name]
            if isinstance(
                optimization_rule, float
            ):  # case we optimize for a numeber eg.1

                best = abs(best_metrics[metric_name]) - optimization_rule
                current = abs(eval_metrics[metric_name]) - optimization_rule
                best_value, best_metrics[metric_name] = min(
                    (best, best_metrics[metric_name]),
                    (current, eval_metrics[metric_name]),
                )
                if best_metrics[metric_name] == eval_metrics[metric_name]:
                    has_improved = True
                else:
                    has_improved = False

            else:  # case we have min/max optimization

                best_metrics[metric_name] = optimization_rule(
                    best_metrics[metric_name], eval_metrics[metric_name]
                )
                if best_metrics[metric_name] == eval_metrics[metric_name]:
                    has_improved = True
                else:
                    has_improved = False

    else:
        has_improved = False

    return has_improved, best_metrics


# ==============================================================================
# Train implementations
# ------------------------------------------------------------------------------


@functools.singledispatch
def train(
    model: Model, train_ds, eval_ds, hp: kt.HyperParameters, log_dir: str, eager: bool
):
    raise NotImplementedError(f"train not implemented for {type(model)}")


# ------------------------------------------------------------------------------
# Tensorflow models
# ------------------------------------------------------------------------------


@train.register(tf.Module)
def train_tf(
    model: Model, train_ds, eval_ds, hp: kt.HyperParameters, log_dir: str, eager: bool
) -> metrics.History:
    logger.info("train hyperparameters: %s", hp.values)

    # hyperparamerters
    early_stopping_patience = hp.Int(
        'early_stopping_patience', 2, 20, sampling='log', default=8
    )
    epochs = hp.Int('num_epochs', 10, 200, sampling='log', default=40)
    learning_rate = hp.Float('learning_rate', 5e-6, 5e-2, sampling='log', default=1e-3)
    optimizer_name = hp.Choice(
        'optimizer', ['Adam', 'Nadam', 'SGD', 'RMSprop'], default='Adam'
    )
    optimizer = getattr(tfk.optimizers, optimizer_name)(lr=learning_rate)

    batch_size = hp.Int('batch_size', 8, 512, sampling='log', default=64)

    # compile train and test loops to TF graph
    run_epoch_fn = _run_epoch if eager else tf.function(_run_epoch)

    metrics = model.metrics()

    starting_epoch = 1

    if log_dir is not None:
        summary_writer_train = tf.summary.create_file_writer(
            os.path.join(log_dir, 'train')
        )
        summary_writer_eval = tf.summary.create_file_writer(
            os.path.join(log_dir, 'eval')
        )

        checkpoint = tf.train.Checkpoint(
            step=tf.Variable(starting_epoch), optimizer=optimizer, model=model
        )
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, os.path.join(log_dir, 'train'), max_to_keep=epochs
        )
        write_embeddings = True
        projector_config = None
    else:
        summary_writer_train = None
        summary_writer_eval = None
        write_embeddings = False
        projector_config = None

    best_epoch = starting_epoch
    best_metrics = None
    for epoch in range(starting_epoch, epochs + 1):
        start_time = time.time()
        # if epoch == 2:
        #     tf.profiler.experimental.start(os.path.join(log_dir, 'train'))
        run_epoch_fn(model, _train_step, optimizer, train_ds, batch_size, metrics)
        # if epoch == 2:
        #     tf.profiler.experimental.stop()
        end_time = time.time()
        metrics.output_metrics(
            epoch=epoch,
            epochs=epochs,
            evaluation=False,
            time=end_time - start_time,
            summary_writer=summary_writer_train,
            logger=logger,
        )

        start_time = time.time()
        run_epoch_fn(model, _eval_step, optimizer, eval_ds, batch_size, metrics)
        end_time = time.time()
        metrics.output_metrics(
            epoch=epoch,
            epochs=epochs,
            evaluation=True,
            time=end_time - start_time,
            summary_writer=summary_writer_eval,
            logger=logger,
        )

        if log_dir is not None:
            checkpoint.step.assign_add(1)
            checkpoint_path = checkpoint_manager.save()

        if write_embeddings and projector_config is None:
            projector_config = _embeddings_config(
                os.path.join(log_dir, 'train'), checkpoint_path, model.vocabularies
            )
            _write_embeddings_config(os.path.join(log_dir, 'train'), projector_config)

        eval_metrics = {
            m: metrics.history.eval[m][-1] for m in model.early_stopping_config.keys()
        }
        has_improved, best_metrics = compare_metrics(
            best_metrics, eval_metrics, model.early_stopping_config
        )

        if has_improved:
            best_epoch = epoch

        elif (
            early_stopping_patience is not None
            and epoch >= best_epoch + early_stopping_patience
        ):
            logger.info(
                f"Early stopping, eval {model.early_stopping_config} did not improve for {early_stopping_patience} epochs."
            )
            break

    if log_dir is not None and epochs > 0 and early_stopping_patience is not None:
        best = checkpoint_manager.checkpoints[-1 - (epoch - best_epoch)]
        checkpoint.restore(best).assert_consumed()
        if projector_config is not None:
            projector_config.model_checkpoint_path = best
            _write_embeddings_config(os.path.join(log_dir, 'train'), projector_config)
        logger.info(f"Restored weights from best epoch {best_epoch}.")

    if early_stopping_patience is not None:
        metrics.history.best_epoch = best_epoch
    else:
        metrics.history.best_epoch = epochs

    return metrics.history


def _plot_outliers(
    idxs: tf.Tensor,
    **kwargs: typing.Dict[str, typing.Union[typing.Dict[str, tf.Tensor], tf.Tensor]],
):
    """Plot outliers at position `idxs` in given tensors (`kwargs`).

    It will flatten one level for dict arguments (and ignore the keyword name).

    Example:
        >>> _plot_outliers(tf.where(loss > 70), loss=loss, Y_pred=Y_pred, X=X, Y=Y)

    """
    tensors = dict(
        itertools.chain.from_iterable(
            val.items() if isinstance(val, dict) else [(key, val)]
            for key, val in kwargs.items()
        )
    )
    if tf.size(idxs) > 0:
        features = {
            name: tf.squeeze(tf.gather_nd(tensor, idxs))
            for name, tensor in tensors.items()
        }
        tf.print('Outlier:', features)


def _run_epoch(model: Model, step_fn, optimizer, ds, batch_size, metrics):
    for i in tf.range(0, len(ds), batch_size):
        X, Y = ds[i : i + batch_size]
        Y_pred, loss = step_fn(model, optimizer, X, Y)
        metrics.update_state(Y, Y_pred, loss)


def _train_step(model: Model, optimizer, X, Y):
    with tf.GradientTape() as tape:
        Y_pred = model(X, training=True)
        loss = model.loss(Y, Y_pred)
        # _plot_outliers(tf.where(loss > 20), loss=loss, Y_pred=Y_pred, X=X, Y=Y)
        custom_losses = _layer_losses(model)
        if len(custom_losses):
            # adding the regularization loss to an n-dimensional
            # (element-wise) rather than a scalar loss doesn't affect
            # the mean value, so we support both
            loss += tf.math.add_n(custom_losses)
        avg_loss = tf.reduce_mean(loss)

    grads = tape.gradient(avg_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return Y_pred, loss


def _eval_step(model: Model, optimizer, X, Y):
    Y_pred = model(X, training=False)
    loss = model.loss(Y, Y_pred)
    # _plot_outliers(tf.where(loss > 20), loss=loss, Y_pred=Y_pred, X=X, Y=Y)
    custom_losses = _layer_losses(model)
    if len(custom_losses):
        loss += tf.math.add_n(custom_losses)
    return Y_pred, loss


def _layer_losses(model):
    losses = list(
        itertools.chain.from_iterable(
            # get all keras layer losses
            # https://www.tensorflow.org/api_docs/python/tf/Module
            map(lambda layer: layer.losses, model.submodules)
        )
    )
    return losses


def _embeddings_config(log_dir, checkpoint_path, vocabularies):
    config = tensorboard.plugins.projector.ProjectorConfig()

    for tensor_name, shape in tf.train.list_variables(checkpoint_path):
        if not tensor_name.endswith('/embeddings/.ATTRIBUTES/VARIABLE_VALUE'):
            continue

        # extract the feature name from the name of the embedding tensor
        name = tensor_name.split('/')[-4]
        # strip _cat suffix used for external string->int mappings
        if (
            name not in vocabularies
            and name.endswith('_cat')
            and name[: -len('_cat')] in vocabularies
        ):
            name = name[: -len('_cat')]
        filename = name + '.tsv'

        embedding = config.embeddings.add()
        embedding.tensor_name = tensor_name
        embedding.metadata_path = os.path.join('embeddings_metadata', filename)

        metadata_full_path = os.path.join(log_dir, embedding.metadata_path)
        os.makedirs(os.path.dirname(metadata_full_path), exist_ok=True)
        with tf.io.gfile.GFile(metadata_full_path, 'w') as f:
            vocabulary = vocabularies[name]
            # write header only for multi-column metadata
            if len(vocabulary) > 1:
                header = list(vocabulary.keys())
                f.write('\t'.join(header) + '\n')
            for items in zip(*vocabulary.values()):
                f.write('\t'.join([str(item) for item in items]) + '\n')
            vocab_len = len(next(iter(vocabulary.values())))
            for i in range(shape[0] - vocab_len):
                f.write('out_of_vocab_{}'.format(i))

    return config


def _write_embeddings_config(log_dir, config):
    with tf.io.gfile.GFile(os.path.join(log_dir, 'projector_config.pbtxt'), 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(config))


# ------------------------------------------------------------------------------
# Prophet models
# ------------------------------------------------------------------------------


@train.register
def train_prophet(
    model: sales_prophet.Model,
    train_ds,
    eval_ds,
    hp: kt.HyperParameters,
    log_dir: str,
    eager: bool,
) -> metrics.History:
    train_df = model._dataset_to_dataframe(train_ds)
    eval_df = model._dataset_to_dataframe(eval_ds)
    start_time = time.time()
    model._fit(train_df, eval_df, hp, log_dir)
    train_pred_df = model._predict(train_df)
    end_time = time.time()

    metrics = model.metrics()

    metrics.update_state(
        train_df.sales_before_returns,
        train_pred_df.predicted_sales_before_returns,
        loss=0,
    )
    metrics.output_metrics(
        epoch=1, epochs=1, evaluation=False, time=end_time - start_time, logger=logger,
    )

    start_time = time.time()
    eval_pred_df = model._predict(eval_df)
    end_time = time.time()
    metrics.update_state(
        eval_df.sales_before_returns,
        eval_pred_df.predicted_sales_before_returns,
        loss=0,
    )
    metrics.output_metrics(
        epoch=1, epochs=1, evaluation=True, time=end_time - start_time, logger=logger,
    )
    metrics.history.best_epoch = 1
    return metrics.history
