import tensorflow as tf
import keras
from keras.layers import *


def lstm_1d_deep_fused(hp):
    """
    Builds and compiles a deep LSTM model with hyperparameter tuning for a sequence input of shape (10, 4).

    Args:
        hp (HyperParameters): Hyperparameters object for tuning.

    Returns:
        keras.Model: Compiled Keras model.
    """

    model = keras.Sequential()
    model.add(LSTM(units=hp.Int('lstm1_units', min_value=16, max_value=64, step=16), 
                   return_sequences=True, input_shape=(10, 4)))
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int('lstm2_units', min_value=32, max_value=128, step=32), return_sequences=True)) 
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int('lstm3_units', min_value=64, max_value=128, step=32), return_sequences=True)) 
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int('lstm4_units', min_value=64, max_value=256, step=32)))
    model.add(BatchNormalization())

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


def lstm_1d_deep(hp):
    """
    Builds and compiles a deep LSTM model with hyperparameter tuning for a sequence input of shape (10, 3).

    Args:
        hp (HyperParameters): Hyperparameters object for tuning.

    Returns:
        keras.Model: Compiled Keras model.
    """
    
    model = keras.Sequential()
    model.add(LSTM(units=hp.Int('lstm1_units', min_value=16, max_value=64, step=16), 
                   return_sequences=True, input_shape=(10, 3)))
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int('lstm2_units', min_value=32, max_value=128, step=32), return_sequences=True)) 
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int('lstm3_units', min_value=64, max_value=128, step=32), return_sequences=True)) 
    model.add(BatchNormalization())

    model.add(LSTM(units=hp.Int('lstm4_units', min_value=64, max_value=256, step=32)))
    model.add(BatchNormalization())

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