# Training script
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf

import config as config
from data.processor import process_dataset
from evaluate import visualize_loss
from models.fully_connected import fully_connected_model
from models.gru import gru_model
from models.lstm import lstm_model
from models.simple_rnn import simple_rnn_model
from data.processor import load_data
from utils.plotting import plot_predictions
import datetime


def train_model(save_dir, name, model, epochs, dataset_train, dataset_val):
    # Train
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='src/logs')
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[tensorboard_callback],
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save('{}/{}.h5'.format(save_dir, name))
    return model, history


def run_experiments():
    for j in config.experiment_features:
        for seq_length in config.sequence_lengths:
            for name in config.models:
                run_experiment(model_name=name,
                               hidden_layers=config.models.get(name)['hidden_layers'],
                               learning_rate=config.models.get(name)['learning_rate'],
                               sequence_length=seq_length,
                               features=config.experiment_features.get(j)
                               )


def run_experiment(model_name,
                   hidden_layers,
                   learning_rate,
                   sequence_length,
                   features
                   ):
    flattend = True if model_name == 'fully_connected' else False
    train_data, train_dataset, val_data, val_dataset, test_data, test_dataset = load_data(features, flattend, sequence_length)

    save_dir = 'experiments/{}-sequence-length-{}-features-{}'.format(model_name, sequence_length, len(features))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for batch in train_data.take(1):
        inputs, targets = batch
        break

    if model_name == 'fully_connected':
        model = fully_connected_model(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            shape=inputs,
            loss=config.loss,
            use_dropout=True
        )
    elif model_name == 'lstm':
        model = lstm_model(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            shape=inputs,
            loss=config.loss,
            use_dropout=True
        )
    elif model_name == 'simple_rnn':
        model = simple_rnn_model(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            shape=inputs,
            loss=config.loss,
            use_dropout=True
        )

    else:
        model = gru_model(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            shape=inputs,
            loss=config.loss,
            use_dropout=True
        )

    trained_model, history = train_model(
        save_dir=save_dir,
        name=model_name,
        model=model,
        epochs=config.epochs,
        dataset_train=train_data,
        dataset_val=val_data
    )
    print('Evaluate on test data')
    results = trained_model.evaluate(test_data, batch_size=32)
    print('test loss, test acc:', results)
    with open('experiments/experiment_results.txt', 'a') as myfile:
        myfile.write('{}, {}-sequence-length-{}-features-{}, {}, {}\n'.format(datetime.datetime.now(), model_name, sequence_length, len(features), results[0], results[1]))
    visualize_loss(history, 'Training and Validation loss', save_dir)

    with open('{}/{}'.format(save_dir, "results.txt"), 'w') as f:
        f.write("test loss, test acc: {}".format(results))
    visualize_loss(history, "Training and Validation loss", save_dir)

    if config.plot_all_predictions:
        plot_predictions(test_data, test_dataset, trained_model, sequence_length, save_dir, True)


if __name__ == "__main__":
    run_experiments()

