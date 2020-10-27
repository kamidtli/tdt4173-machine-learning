from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Activation, GRU, Input

def gru_model(hidden_layers, shape, optimizer, loss, use_dropout=False):
    inputs = Input(shape=(shape.shape[1], shape.shape[2]))
    x = GRU(hidden_layers[0], return_sequences=False)(inputs)
    for i in range(1,len(hidden_layers)):
        x = GRU(hidden_layers[i])(x)
    if use_dropout:
        x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
    return model

