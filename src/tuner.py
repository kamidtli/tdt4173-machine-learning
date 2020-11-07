import kerastuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

import IPython

import config as config
from utils.csv_utils import read_csv
from data.processor import process_dataset


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(config.sequence_length, len(config.features))))
    for i in range(hp.Int('n_layers', min_value=1, max_value=5)):
        hp_units = hp.Int('units'+str(i), min_value=10, max_value=256, step=32)
        model.add(keras.layers.GRU(units=hp_units, activation='relu', return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(1))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss = keras.losses.MeanSquaredError(),
                  metrics = ["mae"])

    return model

tuner = kt.Hyperband(model_builder,
                     objective = 'val_mae',
                     max_epochs = 10,
                     factor = 3,
                     directory = 'tuning_results',
                     project_name = 'gru1')

class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait = True)


train_dataset = read_csv('../data/train_data.csv')
train_data = process_dataset(
    train_dataset,
    features=config.features,
    flattened=False,
    sequence_length=config.sequence_length,
    batch_size=config.batch_size
)

val_dataset = read_csv('../data/validation_data.csv')
val_data = process_dataset(
    val_dataset,
    features=config.features,
    flattened=False,
    sequence_length=config.sequence_length,
    batch_size=config.batch_size
)

test_dataset = read_csv('../data/test_data.csv')
test_data = process_dataset(
    test_dataset,
    features=config.features,
    flattened=False,
    sequence_length=config.sequence_length,
    batch_size=config.batch_size
)

tuner.search(train_data, epochs = 100, validation_data = val_data, callbacks = [ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

optimal_units = [best_hps.get('units' + str(i)) for i in range(best_hps.get('n_layers'))]
print(f"""
The hyperparameter search is complete.\n 
Optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
Optimal units: {optimal_units} 
""")

# {best_hps.get('units')}