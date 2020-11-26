from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Activation, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model, Sequential
import config

def fully_connected_model(hidden_layer, dense_layer, shape, use_dropout=False):
    model = Sequential()
    model.add(Flatten(input_shape=(shape.shape[1], shape.shape[2])))
    model.add(Dense(hidden_layer, activation='relu'))
    model.add(Dense(dense_layer, activation='relu'))
    model.add(Dense(1))
    return model

def fully_connected_model2(hidden_layers, shape, use_dropout=False):
    inputs = Input(shape=(shape.shape[1], shape.shape[2]))
    x = Flatten()(inputs)
    for i in hidden_layers:
        x = Dense(i, activation='relu')(x)
    if use_dropout:
        x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model


def tune_fully_connected(hp):
    model = Sequential()
    model.add(Input(shape=(config.sequence_length, len(config.features))))
    hp_units = hp.Int('units', min_value=10, max_value=256, step=32)
    model.add(Dense(units=hp_units,))
    dn_units = hp.Int('dense_units', min_value=10, max_value=256, step=32)
    model.add(Dense(dn_units, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # Tune the learning rate for the optimizer
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss=config.loss,
                  metrics=["mae"])
    return model


