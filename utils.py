import numpy as np
import tensorflow as tf
import keras

def quantize(data:np.array, scale:float, zero:float) -> np.array:
    return (data / scale + zero).astype(np.int8)

def dequantize(data:np.array, scale:float, zero:float) -> np.array:
    return ((data-zero)*scale).astype(np.float32)

def reduce_mean_last(last:int=1) -> float:
    def accuracy_last(y_true, y_pred) -> float:
        if len(y_true.shape) == 3:
            y_true = tf.reduce_mean(y_true[:,-last:], axis=1)
            y_pred = tf.reduce_mean(y_pred[:,-last:], axis=1)
        y_true_argmax = tf.argmax(y_true, axis=-1)
        y_pred_argmax = tf.argmax(y_pred, axis=-1)
        return tf.reduce_mean(tf.cast(tf.equal(y_true_argmax, y_pred_argmax), tf.float32))
    return accuracy_last


def reduce_flatten() -> float:
    def accuracy_flattened(y_true, y_pred) -> float:
        if len(y_true.shape) == 3:
            new_shape = (-1, tf.shape(y_true)[-1])
            y_true = tf.reshape(y_true, new_shape)
            y_pred = tf.reshape(y_pred, new_shape)
        y_true_argmax = tf.argmax(y_true, axis=-1)
        y_pred_argmax = tf.argmax(y_pred, axis=-1)
        return tf.reduce_mean(tf.cast(tf.equal(y_true_argmax, y_pred_argmax), tf.float32))
    return accuracy_flattened


@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class GestureAccuracy(keras.metrics.MeanMetricWrapper):
    def __init__(self, mean:int=1, name=None, dtype=None, **kwargs):
        self.mean = mean
        self.reduction_fn = reduce_mean_last(mean)
        super().__init__(self.reduction_fn, name, dtype, **kwargs)

    def get_config(self):
        conf = super().get_config()
        conf.update({'mean': self.mean})
        return conf
    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class FrameAccuracy(keras.metrics.MeanMetricWrapper):
    def __init__(self, name=None, dtype=None, **kwargs):
        self.reduction_fn = reduce_flatten()
        super().__init__(self.reduction_fn, name, dtype, **kwargs)

    def get_config(self):
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ref: https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric
@keras.saving.register_keras_serializable(package='TinyssimoRadar')
class ConfusionMatrix(keras.metrics.Metric):
    def __init__(self, class_num, reduction, normalize=False, name=None, dtype=None, **kwargs):
        super().__init__(name, dtype, **kwargs)
        self.class_num = class_num
        self.reduction = reduction

        self.confusion_matrix = self.add_weight(
            name='cm',
            shape=(self.class_num, self.class_num),
            initializer='zeros',
            dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.reduction in ('frames', 'flatten'):
            new_shape = (-1, tf.shape(y_true)[-1])
            y_true = tf.reshape(y_true, new_shape)
            y_pred = tf.reshape(y_pred, new_shape)
        else:
            y_true = tf.reduce_mean(y_true[:,-self.reduction:], axis=1)
            y_pred = tf.reduce_mean(y_pred[:,-self.reduction:], axis=1)

        y_true_argmax = tf.argmax(y_true, axis=-1)
        y_pred_argmax = tf.argmax(y_pred, axis=-1)
        cm = tf.math.confusion_matrix(y_true_argmax, y_pred_argmax, num_classes=self.class_num, dtype=tf.int64)
        self.confusion_matrix.assign_add(cm)

    def result(self):
        return self.confusion_matrix

    def reset_state(self):
        # This function is called between epochs/steps, when a metric is evaluated during training.
        self.confusion_matrix.assign(tf.zeros((self.class_num, self.class_num), dtype=tf.int64))

    def get_config(self):
        config = super().get_config()
        config.update({'class_num': self.class_num})
        config.update({'reduction': self.reduction})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
