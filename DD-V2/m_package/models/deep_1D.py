import tensorflow as tf
import keras
from keras.layers import *


def lstm_1d_deep(hp):
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


def convlstm_1d_deep(hp):
    model = keras.Sequential()
    model.add(ConvLSTM1D(filters=hp.Int('convlstm1_filters',  min_value=16, max_value=64, step=16), 
                         kernel_size=hp.Choice('convlstm1_kernel', values=[5, 7]), 
                         activation='relu',
                         padding="same",
                         data_format='channels_last',
                         dropout=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1), 
                         return_sequences=True,
                         input_shape=(10, 3, 1))) 
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())
    
    model.add(ConvLSTM1D(filters=hp.Int('convlstm2_filters', min_value=32, max_value=128, step=32), 
                         kernel_size=hp.Choice('convlstm2_kernel', values=[5, 7]), 
                         data_format='channels_last',
                         padding="same",
                         dropout=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1), 
                         return_sequences=True,
                         activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
    model.add(BatchNormalization())

    model.add(ConvLSTM1D(filters=hp.Int('convlstm3_filters', min_value=64, max_value=128, step=32), 
                         kernel_size=hp.Choice('convlstm3_kernel', values=[3, 5]), 
                         data_format='channels_last',
                         padding="same",
                         dropout=hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1), 
                         return_sequences=True,
                         activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 2), padding="same"))
    model.add(BatchNormalization())

    model.add(ConvLSTM1D(filters=hp.Int('convlstm4_filters', min_value=64, max_value=256, step=32), 
                         kernel_size=hp.Choice('convlstm4_kernel', values=[2, 3]), 
                         data_format='channels_last',
                         padding="same",
                         dropout=hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.1), 
                         return_sequences=True,
                         activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,2), padding="same"))
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


def conv_1d_deep(hp):
    model = keras.Sequential()
    model.add(Conv1D(filters=hp.Int('filters_1', min_value=16, max_value=64, step=16), 
                     kernel_size=hp.Choice('kernel_size_1', values=[5, 7]), 
                     activation='relu', 
                     input_shape=(10, 3),
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=hp.Int('filters_2', min_value=32, max_value=128, step=32), 
                     kernel_size=hp.Choice('kernel_size_2', values = [5, 7]), 
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=hp.Int('filters_3', min_value=64, max_value=128, step=32), 
                     kernel_size=hp.Choice('kernel_size_3', values = [3, 5]), 
                     activation='relu',
                     padding='same')) 
    model.add(MaxPooling1D(pool_size=2, padding='same'))
    model.add(BatchNormalization())

    model.add(Conv1D(filters=hp.Int('filters_4', min_value=64, max_value=256, step=32), 
                     kernel_size=hp.Choice('kernel_size_4', values = [2, 3]), 
                     activation='relu',
                     padding='same')) 
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2, padding='same'))

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