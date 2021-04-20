# Copyright © 2020 7Learnings GmbH
"""Model utilities"""
import functools
import json
import typing
import importlib

import kerastuner as kt
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.feature_column import sequence_feature_column as sfc

from . import sales_prophet
from .metrics import MetricsBundle


class Model(typing.Protocol):
    def __init__(
        self,
        *,
        vocabularies: typing.Dict[
            str, typing.Dict[str, typing.List[typing.Union[str, float]]]
        ],
        features: typing.Dict[str, tf.io.FixedLenFeature],
        hp: kt.HyperParameters,
        **model_options,
    ):
        ...

    def __call__(self, X: typing.Dict[str, tf.Tensor], training: bool):
        ...

    def loss(self, Y_true: tf.Tensor, Y_pred: tf.Tensor):
        ...

    def metrics(self) -> MetricsBundle:
        ...

    @property
    def trainable_variables(self) -> typing.Sequence[tf.Variable]:
        """Also see: https://www.tensorflow.org/api_docs/python/tf/Module"""
        ...

    vocabularies: typing.Dict[
        str, typing.Dict[str, typing.List[typing.Union[str, float]]]
    ]
    early_stopping_config: typing.Dict[
        str, typing.Union[float, typing.Callable[[float, float], float]]
    ]


def get_model_class(model_type: str) -> typing.Type[Model]:
    """Get Model class for `model_type` by name"""
    try:
        mod = importlib.import_module(f"models.{model_type}")
        if model_clazz := getattr(mod, 'Model', None):
            return model_clazz
    except ImportError:
        pass
    raise ValueError(f"Unimplemented model {model_type}")


def slice_time_series_features(
    X: typing.Dict[str, tf.Tensor], beg: typing.Union[tf.Tensor, int], length: int,
) -> typing.Dict[str, tf.Tensor]:
    """
    Slice each (3 dimensional) time-series tensor in X by beg:beg+length.

    :returns: Dict with sliced tensors.
    """
    X_ret = X.copy()
    for name, t in X_ret.items():
        if t.shape.ndims <= 2:
            continue
        t = t[:, beg : beg + length]
        # set slice to static length for shape compatibility with tensors of known size
        t.set_shape([None, length] + t.get_shape()[2:])
        X_ret[name] = t
    return X_ret


def concat_context_with_time_series_features(
    X: typing.Dict[str, tf.Tensor],
    feature_specs: typing.Dict[str, tf.io.FixedLenFeature],
    lookup_tables: typing.Dict[str, tf.lookup.StaticVocabularyTable],
    embeddings: typing.Dict[str, tfkl.Embedding],
    training: bool,
) -> tf.Tensor:
    """
    Combine time-series and context features in X into single 3 dimensional tensor.
    Context features are repeated over the time dimension.

    :returns: Concatenated 3 dimensional tensor.
    """

    def lookup(table, tensor):
        if training:
            assert tensor.dtype == tf.string, "Only expecting string lookup tables"
            assert table._num_oov_buckets == 1, "Expecting a single OOV bucket"
            # train oov bucket with some random samples
            tensor = tf.where(
                tf.random.uniform(tf.shape(tensor)) >= 0.05,
                tensor,
                "__random_out_of_vocabulary_token",
            )
        return table.lookup(tensor)

    def embed(embedding, tensor):
        # flatten feature * embedding dim (e.g. (128, 28, 2, 6) -> (128, 28, 12))
        shape = (
            [-1]
            + tensor.shape[1:-1].as_list()
            + [tensor.shape[-1] * embedding.output_dim]
        )
        return tf.reshape(embedding(tensor), shape)

    # Convert all features to floats
    # https://github.com/7Learnings/repo/issues/2236
    features = {
        name: embed(embeddings[name], lookup(lookup_tables[name], X[name]))
        if name in embeddings
        else tf.cast(X[name], tf.float32)
        for name, spec in feature_specs.items()
    }
    assert all(
        map(lambda t: t.shape.ndims in [2, 3], features.values())
    ), "Found tensor with Unexpected dimensionality"

    # combine context and time-series features by repeating the former
    context_features = [v for v in features.values() if v.shape.ndims == 2]
    context_features = (
        tfkl.concatenate(context_features)
        if len(context_features) > 1
        else context_features[0]
        if len(context_features)
        else None
    )
    time_series_features = [v for v in features.values() if v.shape.ndims == 3]
    time_series_features = (
        tfkl.concatenate(time_series_features)
        if len(time_series_features) > 1
        else time_series_features[0]
        if len(time_series_features)
        else None
    )
    if context_features is not None and time_series_features is not None:
        return sfc.concatenate_context_input(context_features, time_series_features)
    elif context_features is not None:
        return tf.expand_dims(context_features, axis=1)
    else:
        return time_series_features


def check_expected_feature_shapes(
    feature_specs: typing.Dict[str, tf.io.FixedLenFeature],
    expected_features: typing.Dict[str, typing.List[typing.Optional[int]]],
):
    """
    Check feature_specs for missing or misshaped expected features.

    :raises ValueError: if check fails
    """
    errors = []
    for feat, shape in expected_features.items():
        if feat not in feature_specs:
            errors.append(f"Missing required feature `{feat}`.")
        else:
            actual = tf.TensorShape(feature_specs[feat].shape)
            expected = tf.TensorShape(shape)
            if not actual.is_compatible_with(expected):
                errors.append(
                    f"Shape `{actual}` of feature `{feat}` is not compatible"
                    f"with expected shape `{expected}`"
                )
        if len(errors):
            raise ValueError("Invalid features:\n\t" + "\n\t".join(errors))


def create_embedding_layers(
    feature_specs: typing.Dict[str, tf.io.FixedLenFeature],
    vocabularies: typing.Dict[str, tf.io.FixedLenFeature],
    max_embedding_dimension: int = 6,
) -> typing.Tuple[
    typing.Dict[str, tf.lookup.StaticVocabularyTable], typing.Dict[str, tfkl.Embedding]
]:
    """
    Create category -> int lookup tables and embedding layers for all features in vocabularies.
    """
    lookup_tables = {}
    embeddings = {}
    for name, feature in feature_specs.items():
        if name in vocabularies:
            lookup_tables[name] = _lookup_table(vocabularies[name])
            cardinality = lookup_tables[name].size().numpy()
        elif feature.dtype == tf.string:
            raise ValueError(f"String feature {name} is missing from vocabulary.")
        else:
            continue
        embeddings[name] = tfkl.Embedding(
            cardinality,
            min(cardinality // 2, max_embedding_dimension),
            name=name + '_emb',
        )
    return lookup_tables, embeddings


def _lookup_table(vocabulary):
    return tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            vocabulary, tf.range(len(vocabulary), dtype=tf.int64)
        ),
        num_oov_buckets=1,
    )


def _tensor_specs(
    feature_specs: typing.Dict[str, tf.io.FixedLenFeature],
) -> typing.Dict[str, tf.TensorSpec]:
    """
    Converts tf.io.FixedLenFeature based feature dict to tf.TensorSpec.

    :returns: A new dictionary with the converted tensor specs.
    """
    return {
        name: tf.TensorSpec(shape=[None] + feat.shape, dtype=feat.dtype)
        for name, feat in feature_specs.items()
    }


@functools.singledispatch
def save_model(model, path: str):
    raise NotImplementedError(f"save_model not implemented for {type(model)}")


@save_model.register
def save_model_tf(model: tf.Module, path: str):
    """
    Saves the model's predict function as TF SavedModel format to path.
    Also see tf.saved_model.save
    """
    tf.saved_model.save(
        model,
        f"{path}/model",
        signatures=model.predict.get_concrete_function(**_tensor_specs(model.features)),
    )


@save_model.register
def save_model_prophet(
    model: sales_prophet.Model, path: str,
):
    from fbprophet.serialize import model_to_json

    tf.io.gfile.makedirs(path)
    with tf.io.gfile.GFile(f"{path}/model.json", 'w') as f:
        save_list = []
        for grp, prophet in model.models.items():

            model_dict = {'group': list(grp), 'model': model_to_json(prophet)}
            # TODO: should be saved once for all models rather than with each model
            save_list.append(model_dict)
        json.dump(
            save_list, f,
        )


def load_model_tf(path: str) -> tf.Module:
    return tf.saved_model.load(f"{path}/model")


def load_model_prophet(path: str) -> sales_prophet.Model:
    from fbprophet.serialize import model_from_json

    with tf.io.gfile.GFile(f"{path}/model.json", 'r') as f:
        models = {}
        for elem in json.load(f):
            models[tuple(elem['group'])] = model_from_json(elem['model'])
            # TODO: should be saved once for all models rather than with each model

    return sales_prophet.Model.from_models(models)
