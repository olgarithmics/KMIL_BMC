import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.layers import Layer, multiply

class Mil_Attention(Layer):
    """
    Mil Attention Mechanism

    This layer contains Mil Attention Mechanism

    # Input Shape
        2D tensor with shape: (batch_size, input_dim)

    # Output Shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None,
                    use_bias=True, use_gated=True, **kwargs):
        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.v_init = initializers.get(kernel_initializer)
        self.w_init = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)


        self.v_regularizer = regularizers.get(kernel_regularizer)
        self.w_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Mil_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.V = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)


        self.w = self.add_weight(shape=(self.L_dim, 1),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)


        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True


    def call(self, x, mask=None):
        n, d = x.shape
        ori_x = x
        # do Vhk^T
        x = K.tanh(K.dot(x, self.V)) # (2,64)

        if self.use_gated:
            gate_x = K.sigmoid(K.dot(ori_x, self.U))
            ac_x = x * gate_x
        else:
            ac_x = x

        # do w^T x
        soft_x = K.dot(ac_x, self.w)  # (2,64) * (64, 1) = (2,1)
        alpha = K.tanh(K.transpose(soft_x)) # (2,1)
        alpha = K.transpose(alpha)
        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'v_initializer': initializers.serialize(self.V.initializer),
            'w_initializer': initializers.serialize(self.w.initializer),
            'v_regularizer': regularizers.serialize(self.v_regularizer),
            'w_regularizer': regularizers.serialize(self.w_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Mil_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Graph_Attention(Layer):
    """
    Implementation of the Graph Attention Mechanism

    # Arguments

        L_dim:              dimensionality of the attn_kernel_self matrix
        output_dim:         positive integer, dimensionality of the output space
        kernel_initializer: initializer of the `kernel` weights matrix
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        bias_initializer:   initializer of the `bias` weights
        bias_regularizer:   regularizer function applied to the `bias` weights
        use_gated:          boolean, whether use the gated attenion mechanism or not



    # Input Shape
        2D tensor with shape: (n, input_dim) corresponding to the feature representations h_1,h_2,....,h_n of every bag

    # Output Shape
        2D tensor with shape: (n, n) containing the relevance score between all the instances of a bag either connected or not

    """

    def __init__(self, L_dim, output_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None, use_gated=False,
                 **kwargs):

        self.L_dim = L_dim
        self.output_dim = output_dim
        self.use_gated = use_gated

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.attn_kernel_initializer = initializers.get(kernel_initializer)
        self.neighbor_weight_initializer = initializers.get(kernel_initializer)
        self.u_init = initializers.get(kernel_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.attn_kernel_regularizer = regularizers.get(kernel_regularizer)
        self.neighbor_weight_regularizer = regularizers.get(kernel_regularizer)
        self.u_regularizer = regularizers.get(kernel_regularizer)

        super(Graph_Attention, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2

        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.L_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        self.attn_kernel_self = self.add_weight(shape=(self.L_dim, 1),
                                                initializer=self.attn_kernel_initializer,
                                                name='attn_kernel_self',
                                                regularizer=self.attn_kernel_regularizer,
                                                trainable=True)

        self.attn_kernel_neighs = self.add_weight(shape=(self.L_dim, 1),
                                                  initializer=self.neighbor_weight_initializer,
                                                  name='attn_kernel_neigh',
                                                  regularizer=self.neighbor_weight_regularizer,
                                                  trainable=True)

        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.L_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True

    def call(self, input_tensor, mask=None):
        X = input_tensor

        x = K.tanh(K.dot(X, self.kernel))

        if self.use_gated:
            gate_x = K.sigmoid(K.dot(X, self.U))
            ac_x = x * gate_x
        else:
            ac_x = x

        attn_self = K.dot(ac_x, self.attn_kernel_self)

        attn_for_neighs = K.dot(ac_x, self.attn_kernel_neighs)

        data_input = attn_self + K.transpose(attn_for_neighs)

        data_input = K.tanh(data_input)

        return data_input

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.self.kernel_initializer),
            'attn_kernel_self': initializers.serialize(self.attn_kernel_self),
            'u_init': initializers.serialize(self.u_init),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'u_regularizer': regularizers.serialize(self.u_regularizer),
            'use_bias': self.use_bias,
            "use_gated": self.use_gated
        }
        base_config = super(Graph_Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class NeighborAggregator(Layer):
    """
    Aggregation of neighborhood information

    This layer is responsible for aggregatting the neighborhood information of the attentin matrix through the
    element-wise multiplication with an adjacency matrix. Every row of the produced
    matrix is averaged to produce a single attention score.

    # Arguments
        output_dim:            positive integer, dimensionality of the output space

    # Input shape
        2D tensor with shape: (n, n)
        2d tensor with shape: (None, None) correspoding to the adjacency matrix
    # Output shape
        2D tensor with shape: (1, units) corresponding to the attention coefficients of every instance in the bag
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim

        super(NeighborAggregator, self).__init__(**kwargs)

    def sparse_mean(self, sparse_tensor, non_zero_elements):
        reduced_sum = tf.sparse.reduce_sum(sparse_tensor, 1)
        reduced_mean = tf.math.divide(
            reduced_sum, non_zero_elements, name=None)
        return reduced_mean

    def call(self, input_tensor, mask=None):
        data_input = input_tensor[0]

        adj_matrix = input_tensor[1]

        data_input = multiply([adj_matrix, data_input])

        non_zero_elements = tf.cast(tf.math.count_nonzero(adj_matrix, 1), tf.float32)
        sparse = tf.sparse.from_dense(data_input)

        sparse_mean = self.sparse_mean(sparse, non_zero_elements)
        tensor_vector = tf.reshape(tensor=sparse_mean, shape=(tf.shape(data_input)[1],))

        alpha = K.softmax(K.transpose(tensor_vector))
        alpha = K.transpose(alpha)

        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)


class Last_Sigmoid(Layer):
    """
    Attention Activation

    This layer contains the last sigmoid layer of the network


    # Arguments
        output_dim:         positive integer, dimensionality of the output space
        kernel_initializer: initializer of the `kernel` weights matrix
        bias_initializer:   initializer of the `bias` weights
        kernel_regularizer: regularizer function applied to the `kernel` weights matrix
        bias_regularizer:   regularizer function applied to the `bias` weights
        use_bias:           boolean, whether use bias or not

    # Input shape
        2D tensor with shape: (n, input_dim)
    # Output shape
        2D tensor with shape: (1, units)
    """

    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(Last_Sigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        x = K.sum(x, axis=0, keepdims=True)
        x = K.dot(x, self.kernel)
        if self.use_bias:
            x = K.bias_add(x, self.bias)
        out = K.sigmoid(x)
        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'kernel_initializer': initializers.serialize(self.kernel.initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'use_bias': self.use_bias
        }
        base_config = super(Last_Sigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DistanceLayer(Layer):

    def __init__(self, output_dim=1, **kwargs):
        self.output_dim = output_dim

        super(DistanceLayer, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        x, y = input_tensor
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)

        return K.sqrt(K.maximum(sum_square, K.epsilon()))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(DistanceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
