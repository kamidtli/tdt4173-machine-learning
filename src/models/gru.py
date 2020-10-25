from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Activation, GRU, Input

def gru_model(hidden_layers, shape, optimizer, loss, use_dropout=False):
    inputs = x = Input(shape=(shape.shape[1], shape.shape[2]))
    print(hidden_layers)
    for i in hidden_layers:
        x = GRU(i)(x)
    if use_dropout:
        inputs = Dropout(0.5)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model

