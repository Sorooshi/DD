import tensorflow as tf
import keras
from keras.layers import *

def conv_2d_deep(hp):
    model = keras.Sequential()
    model.add(Conv2D(filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),
                     kernel_size=hp.Choice('conv1_kernel', values = [5, 7]),
                     activation='relu',
                     input_shape=(60, 180, 1)))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=hp.Int('conv2_filters', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Choice('conv2_kernel', values = [5, 7]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=hp.Int('conv3_filters', min_value=64, max_value=128, step=32),
                     kernel_size=hp.Choice('conv3_kernel', values = [3, 5]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=hp.Int('conv4_filters', min_value=64, max_value=256, step=32),
                     kernel_size=hp.Choice('conv4_kernel', values = [2, 3]),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu'))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(hp.Int('dense_units_2', min_value=32, max_value=128, step=32), activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(2, activation='sigmoid'))
    
    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    
    optimizer_name = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name='auc')],
    )
    return model


def conv_2d_basic(hp):
    model = keras.Sequential()
    model.add(Conv2D(filters=hp.Int('filters_1', min_value=32, max_value=128, step=32), 
                     kernel_size=(hp.Choice('kernel_size_1', values=[5, 7])), 
                     activation='relu', 
                     input_shape=(60, 180, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=hp.Int('filters_2', min_value=64, max_value=256, step=32), 
                     kernel_size=(hp.Choice('kernel_size_2', values=[3, 5])), 
                     activation='relu')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))


    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    
    optimizer_name = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name='auc')],
    )
    return model