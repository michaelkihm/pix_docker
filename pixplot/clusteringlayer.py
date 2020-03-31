#import tensorflow as tf 
from tensorflow import keras



class ClusteringLayer(keras.layers.Layer):
    """
    @briref Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
            sample belonging to each cluster. The probability is calculated with student's t-distribution.
    @param n_clusters: number of clusters.
    @param weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
    @param alpha: parameter in Student's t-distribution. Default to 1.0.
    @return 2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = keras.layers.InputSpec(ndim=2)
        self.clusters = None

    def build(self, input_shape):
        """
        @brief build clustering layer
        """
        assert len(input_shape) == 2
        input_dim = input_shape.as_list()[1]
        self.input_spec = keras.layers.InputSpec(
            dtype=keras.backend.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(
            self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        
        super().build(input) #must be at the end

    def call(self, inputs):
        """ 
        @brief compute t-distribution:  q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        @param inputs: the variable containing data, shape=(n_samples, n_features)
        @return q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (keras.backend.sum(keras.backend.square(keras.backend.expand_dims(inputs,
                                                       axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = keras.backend.transpose(keras.backend.transpose(q) / keras.backend.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
