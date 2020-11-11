from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, Input
import config


def simple_rnn_model(hidden_layer,dense_layer, shape, use_dropout=False):
    model = Sequential()
    model.add(SimpleRNN(hidden_layer, input_shape=(shape.shape[1], shape.shape[2])))
    model.add(Dense(dense_layer, activation='relu'))
    if use_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(1))
    return model

def simple_rnn_model2(hidden_layers, shape, use_dropout=False):
    inputs = x = Input(shape=(shape.shape[1], shape.shape[2]))
    for i in hidden_layers:
        x = SimpleRNN(i, return_sequences=True)(x)
    x = Dense(64, activation='relu')(x)
    if use_dropout:
        x = Dropout(0.5)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    return model

def tune_simple_rnn(hp):
    model = Sequential()
    model.add(Input(shape=(config.sequence_length, len(config.features))))
    hp_units = hp.Int('units', min_value=10, max_value=256, step=32)
    model.add(SimpleRNN(units=hp_units))
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


