from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, TimeDistributed, Dense, Activation, LSTM, Input

def lstm_model(hidden_layers, shape, optimizer, loss, use_dropout=False):
    inputs = Input(shape=(shape.shape[1], shape.shape[2]))
    x = LSTM(hidden_layers[0], return_sequences=False)(inputs)
    for i in range(1,len(hidden_layers)):
        x = LSTM(hidden_layers[i])(x)
    if use_dropout:
        inputs = Dropout(0.5)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model

