"""Main module."""

from math import pi
from tensorflow import keras

import time
import numpy
from typing import List, Union, Callable

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


class BaseAutoEncoder(object):
    def __init__(self):
        self.model = None

    @staticmethod
    def circular_unit(name: str, comp: int = 2) -> Callable:
        def func(x):
            out = []
            if comp < 2:
                raise ValueError("comp must be at least 2")
            elif comp == 2:
                out = [keras.layers.Lambda(lambda x: keras.backend.sin(x), name=name + '_sin')(x),
                       keras.layers.Lambda(lambda x: keras.backend.cos(x), name=name + '_cos')(x)]
            else:
                out = [
                    keras.layers.Lambda(lambda x: keras.backend.sin(x + 2 * pi * i / comp), name=name + '_' + str(i))(x)
                    for i in range(comp)]
            out = keras.layers.Concatenate(name=name + '_out')(out)
            return out

        return func

    @staticmethod
    def logistic_unit(name: str, n: int, trans: bool = True, reg_scale: float = 1e-2,
                      reg_trans: float = 1e-2) -> Callable:
        def func(x):
            x = keras.layers.Dense(name=name + '_scale',
                                   units=n,
                                   use_bias=trans,
                                   kernel_regularizer=keras.regularizers.l2(reg_scale),
                                   bias_regularizer=keras.regularizers.l2(reg_trans),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                   bias_initializer=keras.initializers.Zeros()
                                   )(x)
            x = keras.layers.Activation(name=name + '_out',
                                        activation='tanh'
                                        )(x)
            return x

        return func

    @staticmethod
    def linear_unit(name: str, n: int, trans: bool = True, reg_scale: float = 1e-2,
                    reg_trans: float = 1e-2) -> Callable:
        def func(x):
            x = keras.layers.Dense(name=name + '_scale',
                                   units=n,
                                   use_bias=trans,
                                   kernel_regularizer=keras.regularizers.l2(reg_scale),
                                   bias_regularizer=keras.regularizers.l2(reg_trans),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                   bias_initializer=keras.initializers.Zeros()
                                   )(x)
            return x

        return func

    @staticmethod
    def encoder(name: str, size: List[int], reg: float, drop: float, act: Union[str, Callable] = 'tanh') -> Callable:
        def func(x):
            for i, w in enumerate(size):
                x = keras.layers.Dense(name=name + str(i) + '_scale',
                                       units=w,
                                       kernel_regularizer=keras.regularizers.l2(reg),
                                       kernel_initializer=keras.initializers.glorot_normal(seed=None),

                                       )(x)
                if drop > 0:
                    x = keras.layers.Dropout(name=name + str(i) + '_dropout',
                                             rate=drop
                                             )(x)
                x = keras.layers.Activation(name=name + str(i) + '_act',
                                            activation=act
                                            )(x)

            x = keras.layers.Dense(name=name + '_out',
                                   units=1,
                                   use_bias=False,
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None)
                                   )(x)
            return x

        return func

    @staticmethod
    def linear_bypass(name: str, n: int, reg: float) -> Callable:
        def func(x):
            x = keras.layers.Dense(name=name + '_out',
                                   units=n,
                                   use_bias=True,
                                   kernel_regularizer=keras.regularizers.l2(reg),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None)
                                   )(x)

            return x

        return func

    @staticmethod
    def my_sigmoid_gate_bypass(name: str, n: int, reg: float) -> Callable:

        def func(x):
            x = keras.layers.Dense(name=name + '_act',
                                   units=n,
                                   use_bias=True,
                                   kernel_regularizer=keras.regularizers.l2(reg),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None)


                                   )(x)
            x = keras.layers.Activation(name=name + '_out',
                                        activation="sigmoid"
                                        )(x)

            return x

        return func

    @staticmethod
    def linear_decoder(name: str, n: int, reg: float) -> Callable:
        def func(x):
            x = keras.layers.Dense(name=name + 'decode_out',
                                   units=n,
                                   use_bias=True,
                                   kernel_regularizer=keras.regularizers.l2(reg),
                                   kernel_initializer=keras.initializers.glorot_normal(seed=None),

                                   )(x)
            x = keras.layers.Activation(name=name + 'sigmoid_out',
                                        activation="sigmoid"
                                        )(x)
            return x

        return func

    @staticmethod
    def decoder(name: str, n: int) -> Callable:

        def func(x: list):
            if len(x) > 1:
                x = keras.layers.Concatenate(name=name + '_concat')(x)
            else:
                x = x[0]
            x = keras.layers.Dense(name=name + '_act',
                                   units=n,
                                   use_bias=False,
                                   kernel_initializer=keras.initializers.Zeros()
                                   )(x)

            return x

        return func


    def load(self, filepath):
        """Load a BaseAutoEncoder object

        :param filepath:
        :return:
        """
        self.model = keras.models.load_model(filepath, custom_objects={"keras": keras, "keras.backend": keras.backend})

    class MyCallback(keras.callbacks.Callback):

        def __init__(self, interval):
            super().__init__()
            self.cnt = 0
            self.interval = interval
            self.start_time = 0
            self.rec = {'time': [], 'loss': []}

        def on_train_begin(self, logs=None):
            self.start_time = time.time()

        def on_epoch_end(self, batch, logs=None):
            self.cnt += 1
            self.rec['time'].append(time.time() - self.start_time)
            self.rec['loss'].append(logs.get('loss'))
            if self.cnt % self.interval == 0:
                print(f'epoch: {self.cnt}/{self.params["epochs"]}, loss: {logs.get("loss") : .4f}, '
                      f'time elapsed: {self.rec["time"][-1] : .2f}s, '
                      f'time left: {((self.params["epochs"] / self.cnt - 1) * self.rec["time"][-1]) : .2f}s')

    def show_structure(self):
        """Show the structure of the network

        :return: The graph for the structure
        """
        from IPython.display import SVG
        from keras.utils.vis_utils import model_to_dot
        return SVG(model_to_dot(self.model, show_shapes=True).create(prog='dot', format='svg'))


def my_k_sparsity(encoded, sparsity, batch_size, hidden_units) -> Callable:
    def _interleave(xs, ys):
        """Interleaves the two given lists (assumed to be of equal length)."""
        return [val for pair in zip(xs, ys) for val in pair]

    encoded_t = tf.transpose(encoded)  # tf.transpose()数组转置
    print("encoded_t:", encoded_t)
    k = int(sparsity * batch_size)
    _, top_indices = tf.nn.top_k(encoded_t, k=k, sorted=False)

    top_k_unstacked = tf.unstack(top_indices, axis=1)
    row_indices = [tf.range(hidden_units) for _ in range(k)]
    combined_columns = tf.transpose(tf.stack(_interleave(row_indices, top_k_unstacked)))
    indices = tf.reshape(combined_columns, [-1, 2])
    indices = tf.cast(indices, dtype=tf.int32)

    updates = tf.ones(hidden_units * k)
    shape = tf.constant([hidden_units, batch_size])
    print("indices:", indices)
    print("updates:", updates)
    mask = tf.scatter_nd(indices, updates, shape)

    # sparse_encoded = encoded * tf.transpose(mask)
    def funnc(x):
        sparse_encoded = keras.layers.Lambda(lambda x: x * tf.transpose(mask), name="k_spare" + '_out')(x)

        return sparse_encoded

    return funnc


class AutoEncoder(BaseAutoEncoder):


    def __init__(self,
                 input_width: int = None,
                 encoder_depth: int = 2,
                 encoder_width: Union[int, List[int]] = 50,
                 n_circular_unit: int = 1,
                 n_linear_bypass: int = 0,
                 dropout_rate: float = 0.0,
                 nonlinear_reg: float = 1e-4,
                 linear_reg: float = 1e-4,
                 filepath: str = None
                 ):

        super().__init__()
        self.n_circular_unit = n_circular_unit
        if input_width is None:
            self.load(filepath)
        else:
            if type(encoder_width) is int:
                encoder_size = [encoder_width] * encoder_depth
            elif type(encoder_width) is list and len(encoder_width) == encoder_depth:
                encoder_size = encoder_width
            else:
                raise ValueError(
                    "encoder_width must be either (1) an integer or (2) a list of integer, whose length is "
                    "equal to encoder_depth.")


            y = keras.Input(shape=(input_width,), name='input')
            print("y:",y)

            x = self.encoder('encoder', encoder_size, nonlinear_reg, dropout_rate, 'tanh')(y)

            chest = []

            if n_linear_bypass > 0:
                # hidden = 100
                hidden = 200
                # batch = 19 # PC3
                # batch = 16 # 416b
                # batch = 10
                 batch = 32
                # batch=35 # Quartz
                # batch = 50 # 模拟数据
                # batch = 31 # hMyo
                # batch = 47
                # batch = 19 # mouse_ES
                # batch = 37 #mouse_ES
                # batch = 69
                k = 0.1

                x_bypass = self.my_sigmoid_gate_bypass('sigmoid_gate', hidden, linear_reg)(y)

                x_bypass_k = my_k_sparsity(x_bypass, k, batch, hidden)(x_bypass)

                x_sigmoid_decode = self.linear_decoder("sigmoid_decode", y.shape[1], linear_reg)(x_bypass_k)

                x_linear_bypass = self.linear_bypass("linear", 1, linear_reg)(x_sigmoid_decode)
                print("x_linear_bypass:",x_linear_bypass)

                chest.append(x_linear_bypass)

            if n_circular_unit > 0:
                x_circular = self.circular_unit('circular')(x)
                chest.append(x_circular)

            y_hat = self.decoder('decoder', input_width)(chest)


            self.model = keras.Model(outputs=y_hat, inputs=y)

    def train(self, data, batch_size: int = None, epochs: int = 100, verbose: int = 10, rate: float = 1e-4):
        self.model.compile(loss='mean_squared_error',
                           optimizer=keras.optimizers.Adam(rate))
        my_callback = self.MyCallback(verbose)
        print("batch_size:",batch_size)
        # history = self.model.fit(data, data, batch_size=31, epochs=epochs, verbose=0, callbacks=[my_callback])
         history = self.model.fit(data, data, batch_size=32, epochs=epochs, verbose=0, callbacks=[my_callback])
        # history = self.model.fit(data, data, batch_size=16, epochs=epochs, verbose=0, callbacks=[my_callback])
        #history = self.model.fit(data, data, batch_size=47, epochs=epochs, verbose=0, callbacks=[my_callback])
        # history = self.model.fit(data, data, batch_size=35, epochs=epochs, verbose=0, callbacks=[my_callback])
        # history = self.model.fit(data, data, batch_size=19, epochs=epochs, verbose=0, callbacks=[my_callback])
        # history = self.model.fit(data, data, batch_size=50, epochs=epochs, verbose=0, callbacks=[my_callback])
        # history = self.model.fit(data, data, batch_size=37, epochs=epochs, verbose=0, callbacks=[my_callback])
        # history = self.model.fit(data, data, batch_size=69, epochs=epochs, verbose=0, callbacks=[my_callback])
        # history = self.model.fit(data, data, batch_size=batch_size, epochs=epochs, verbose=0, callbacks=[my_callback])
        return history

    def predict_pseudotime(self, data):
        """Predict the circular pseudotime

        :param data: data to be used for training
        :return: the circular pseudotime
        """
        # 获取encoder_out层得输出
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('encoder_out').output]
                                     )([data])

        return res[0]

    def predict_linear_bypass(self, data):
        """Predict the linear bypass loadings.
        """
        # 获取bypass_out层得输出
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('linear_out').output]
                                     )([data])

        return res[0]
    def get_sin(self, data):
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('circular_sin').output]
                                     )([data])
        return res[0]
    def get_cos(self, data):
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('circular_cos').output]
                                     )([data])
        return res[0]
    def get_circular_loadings(self):
        return self.model.get_layer("decoder_act").get_weights()[0][-(self.n_circular_unit * 2):, :]
    def get_circular_component(self, circular_pseudotime):
        if self.n_circular_unit == 1:
            return numpy.hstack([numpy.sin(circular_pseudotime),
                                 numpy.cos(circular_pseudotime)]) @ self.get_circular_loadings()
        else:
            temp = []
            for i in range(self.n_circular_unit):
                temp.append(numpy.sin(circular_pseudotime[:, [i]]))
                temp.append(numpy.cos(circular_pseudotime[:, [i]]))
            return numpy.hstack(temp) @ self.get_circular_loadings()
    def get_circular_out(self, data):
        res = keras.backend.function(inputs=[self.model.get_layer('input').input],
                                     outputs=[self.model.get_layer('circular_out').output]
                                     )([data])
        return res[0]

    def get_weight(self):
        """Get the weight of the transform, where the last two dimensions are for the sinusoidal unit

        :return: a matrix
        """
        layer = self.model.get_layer('decoder_out')
        print("layer:",layer)
        return layer.get_weights()[0]
