import tensorflow as tf
import tensorflow_probability as tfp

# Source: https://gist.github.com/mhorlacher/8599d1204a48caa172e9ecfc0c73b989


class PearsonCorrelation(tf.keras.metrics.Mean):
    def __init__(self, post_proc_fn=None, **kwargs):
        """Pearson Correlation Coefficient.
        Args:
            post_proc_fn (function, optional): Post-processing function for predicted values. Defaults to None.
        """

        super().__init__(name='pearson_correlation', **kwargs)

        self.post_proc_fn = post_proc_fn
        if self.post_proc_fn is None:
            self.post_proc_fn = lambda y, y_pred: (y, y_pred)

    def _compute_correlation(self, y, y_pred):
        corr = tfp.stats.correlation(y, y_pred, sample_axis=1, event_axis=-1)
        return corr

    def _nan_to_zero(self, x):
        is_not_nan = tf.math.logical_not(tf.math.is_nan(x))
        is_not_nan = tf.cast(is_not_nan, tf.float32)

        return tf.math.multiply_no_nan(x, is_not_nan)

    def update_state(self, y, y_pred, **kwargs):
        y, y_pred = self.post_proc_fn(y, y_pred)

        corr = self._compute_correlation(y, y_pred)
        corr = tf.squeeze(corr)

        # remove any nan's that could have been created, e.g. if y or y_pred is a 0-vector
        corr = self._nan_to_zero(corr)

        # assert that there are no inf's or nan's
        tf.debugging.assert_all_finite(
            corr, f'expected finite tensor, got {corr}')

        super().update_state(corr, **kwargs)
