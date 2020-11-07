from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN, Input


def simple_rnn_model(hidden_layers, shape, learning_rate, loss, use_dropout=False):
    inputs = x = Input(shape=(shape.shape[1], shape.shape[2]))
    for i in hidden_layers:
        x = SimpleRNN(i, return_sequences=True)(x)
    if use_dropout:
        x = Dropout(0.5)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model


