from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Activation, GRU, Input
import config

def gru_model2(hidden_layers, shape, use_dropout=False):
    inputs = Input(shape=(shape.shape[1], shape.shape[2]))
    x = GRU(hidden_layers[0])(inputs)
    for i in range(1, len(hidden_layers)):
        x = GRU(hidden_layers[i], return_sequences=True)(x)
    x = Dense(64, activation='relu')(x)
    if use_dropout:
        x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

def gru_model(hidden_layer,dense_layer, shape, use_dropout=False):
    model = Sequential()
    model.add(GRU(hidden_layer, input_shape=(shape.shape[1], shape.shape[2])))
    model.add(Dense(dense_layer, activation='relu'))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(1))
    return model


def tune_gru(hp):
    model = Sequential()
    model.add(Input(shape=(config.sequence_length, len(config.features))))
    hp_units = hp.Int('units', min_value=10, max_value=256, step=32)
    model.add(GRU(units=hp_units))
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
