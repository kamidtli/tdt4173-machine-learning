import kerastuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

import IPython

import config
from utils.csv_utils import read_csv
from data.processor import load_data


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(7, 9)))
    #hp_layers = hp.Int('layers',min_value=1, max_value=5)
    hp_units = hp.Int('units', min_value=10, max_value=256, step=32)
    model.add(keras.layers.GRU(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(1))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss = keras.losses.MeanSquaredError(),
                  metrics = ["mean_squared_error", "accuracy"])

    return model

tuner = kt.Hyperband(model_builder,
                     objective = 'mean_squared_error',
                     max_epochs = 10,
                     factor = 3,
                     directory = 'tuning_results',
                     project_name = 'gru')

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)

df = read_csv('../data/processed/processed_data.csv')
df = df.fillna(-1)
selected = [config.features[i] for i in config.selected_features]
dataset_train, dataset_val = load_data(df, selected, config, normalize_values=False)

tuner.search(dataset_train, epochs = 10, validation_data = dataset_val, callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

#optimal_units = [best_hps.get('units' + str(i)) for i in range(best_hps.get('layers'))]
print(f"""
The hyperparameter search is complete.\n 
Optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
Optimal units: {best_hps.get('units')}
""")