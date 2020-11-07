from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Activation, GRU, Input


def gru_model(hidden_layers, shape, learning_rate, loss, use_dropout=False):
    inputs = Input(shape=(shape.shape[1], shape.shape[2]))
    x = GRU(hidden_layers[0], return_sequences=True)(inputs)
    for i in range(1, len(hidden_layers)):
        x = GRU(hidden_layers[i], return_sequences=True)(x)
    if use_dropout:
        x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model
