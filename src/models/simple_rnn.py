from keras import Sequential
from keras.layers import Dropout, TimeDistributed, Dense, Activation, SimpleRNN


def simple_rnn_model(hidden_layers, use_dropout=False):
    model = Sequential()
    for i in hidden_layers:
        model.add(SimpleRNN(i, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(10)))
    model.add(Activation('softmax'))
    return model

