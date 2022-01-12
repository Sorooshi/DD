import sys
import tensorflow as tf
import tensorflow_probability as tfp


tfk = tf.keras
tfkl = tf.keras.layers
tf.keras.backend.set_floatx('float32')


class VNNRegression(tfk.Model):
    def __init__(self, n_units, input_dim, output_dim, name="vnn_reg", **kwargs):
        super(VNNRegression, self).__init__()

        self.n_units = n_units
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense_1 = tfkl.Dense(units=self.n_units,
                                  activation=tf.nn.relu,
                                  input_shape=(self.input_dim, ))

        self.dense_2 = tfkl.Dense(units=self.output_dim, )

    def call(self, inputs, training=None):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x


class DNNRegression(tfk.Model):
    def __init__(self, n_units, input_dim, output_dim, name="dnn_reg", **kwargs):
        super(DNNRegression, self).__init__()

        self.n_units = n_units
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense_1 = tfkl.Dense(units=self.n_units,
                                  activation=tf.nn.relu,
                                  input_shape=(self.input_dim,))

        self.dense_2 = tfkl.Dense(units=int(2*self.n_units),
                                  activation=tf.nn.relu,)

        self.dropout = tfkl.Dropout(0.3)

        self.dense_3 = tfkl.Dense(units=int(self.n_units),
                                  activation=tf.nn.relu,)

        self.dense_4 = tfkl.Dense(units=self.output_dim)

    def call(self, inputs, training):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dropout(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        return x


class RBFKernelFn(tfkl.Layer):

    def __init__(self, **kwargs):
        super(RBFKernelFn, self).__init__(**kwargs)

        dtype = kwargs.get('dtype', None)

        self._amplitude = self.add_weight(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')

        self._length_scale = self.add_weight(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name="length_scale")

    def call(self, x):
        # never will be called:
        # it just holds the data so that Keras will understand
        return x

    @property
    def kernel(self):
        return tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude=tf.nn.softplus(0.1 * self._amplitude),
            length_scale=tf.nn.softplus(5. * self._length_scale),
        )


# TF loss function:
def determine_reg_tf_loss(loss):

    loss = loss.lower()

    if loss == "mae":
        loss_fn = tfk.losses.mean_absolute_error

    elif loss == "mse":
        loss_fn = tfk.losses.mean_squared_error

    elif loss == "msle":
        loss_fn = tfk.losses.mean_squared_logarithmic_error

    elif loss == "mape":
        loss_fn = tfk.losses.mean_absolute_percentage_error

    elif loss == "kld":
        loss_fn = tfk.losses.kl_divergence

    elif loss == "cosine_similarity":  # check the loss function here
        loss_fn = tfk.losses.cosine_similarity

    elif loss == "squared_hinge":  # check the loss function here
        loss_fn = tfk.losses.SquaredHinge

    else:
        print("Loss function is not defined.")

    return loss_fn


def determine_cls_tf_loss(loss):

    loss = loss.lower()

    if loss == "mae":
        loss_fn = tfk.losses.mean_absolute_error

    else:
        print("Loss function is not defined.")

    return loss_fn


def compile_and_fit(model, optimizer, loss, learning_rate, batch_size,
                    n_epochs, x_train, y_train, x_val, y_val, ):

    if optimizer.lower() == "adam":
        model.compile(optimizer=tfk.optimizers.Adam(learning_rate=learning_rate), loss=loss)

    elif optimizer.lower() == "adamax":
        model.compile(optimizer=tfk.optimizers.Adamax(learning_rate=learning_rate), loss=loss)

    elif optimizer.lower() == "rmsprop":
        model.compile(optimizer=tfk.optimizers.RMSprop(learning_rate=learning_rate), loss=loss)

    elif optimizer.lower() == "sgd":
        model.compile(optimizer=tfk.optimizers.SGD(learning_rate=learning_rate), loss=loss)

    else:
        print("undefined optimizer.")

    history = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                        batch_size=batch_size, epochs=n_epochs, verbose=True,)

    return model, history
