import kerastuner as kt
import tensorflow as tf
from tensorflow import keras
import IPython

import config as config
from models.fully_connected import tune_fully_connected
from models.lstm import tune_lstm
from models.simple_rnn import tune_simple_rnn
from data.processor import load_data
from models.gru import tune_gru

train_data, train_dataset, val_data, val_dataset, test_data, test_dataset = load_data(config.features,
                                                                                      config.sequence_length)


def tune_model(model_builder):
    save_dir = '../tuning_results'
    temp = model_builder.__name__.split("_")
    model = "_".join(temp[1:])
    tuner = kt.Hyperband(model_builder,
                         objective='val_mae',
                         max_epochs=10,
                         factor=3,
                         directory=save_dir,
                         project_name=model)

    tuner.search(train_data, epochs=100, validation_data=val_data, callbacks=[ClearTrainingOutput()])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    with open('{}/{}'.format(save_dir, "tuning_results.txt"), 'a') as f:
        f.write(f""" 
{str.upper(model)}
Optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
Optimal hidden units: {best_hps.get('units')} 
Dense units: {best_hps.get('dense_units')} \n"""
                )


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


def tune_models():
    tuning_models = [tune_gru, tune_lstm, tune_simple_rnn, tune_fully_connected]
    for model in tuning_models:
        tune_model(model)


if __name__ == "__main__":
    tune_models()
