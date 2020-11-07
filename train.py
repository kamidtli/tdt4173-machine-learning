# Training script
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

from src.data.processor import process_dataset
import src.config as config
from src.evaluate import visualize_loss
from src.models.fully_connected import fully_connected_model
from src.models.gru import gru_model
from src.models.simple_rnn import simple_rnn_model
from src.models.lstm import lstm_model
from src.utils.csv_utils import read_csv


def train_model(save_dir, name, model, epochs, dataset_train, dataset_val):
    # Train
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="src/logs")
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[tensorboard_callback],
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save("{}/{}.h5".format(save_dir, name))
    return model, history


def run_experiments():
    for j in config.experiment_features:
        for seq_length in config.sequence_lengths:
            for name in config.models:
                run_experiment(model_name=name,
                               hidden_layers=config.models.get(name)["hidden_layers"],
                               learning_rate=config.models.get(name)["learning_rate"],
                               sequence_length=seq_length,
                               features=config.experiment_features.get(j)
                               )


def run_experiment(model_name,
                   hidden_layers,
                   learning_rate,
                   sequence_length,
                   features
                   ):
    flattend = True if model_name == "fully_connected" else False
    train_dataset = read_csv('data/train_data.csv')
    train_data = process_dataset(
        train_dataset,
        features=features,
        flattened=flattend,
        sequence_length=sequence_length,
        batch_size=config.batch_size
    )

    val_dataset = read_csv('data/validation_data.csv')
    val_data = process_dataset(
        val_dataset,
        features=features,
        flattened=flattend,
        sequence_length=sequence_length,
        batch_size=config.batch_size
    )

    test_dataset = read_csv('data/test_data.csv')
    test_data = process_dataset(
        test_dataset,
        features=features,
        flattened=flattend,
        sequence_length=sequence_length,
        batch_size=config.batch_size
    )

    save_dir = "experiments/{}-sequence-length-{}-features-{}".format(model_name,sequence_length, len(features))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for batch in train_data.take(1):
        inputs, targets = batch
        break

    if model_name == "fully_connected":
        model = fully_connected_model(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            shape=inputs,
            loss=config.loss,
            use_dropout=True
        )
    elif model_name == "lstm":
        model = lstm_model(
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            shape=inputs,
            loss=config.loss,
            use_dropout=True
        )
    elif model_name == "simple_rnn":
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
    print("Evaluate on test data")
    results = trained_model.evaluate(test_data, batch_size=32)
    print("test loss, test acc:", results)
    visualize_loss(history, "Training and Validation loss", save_dir)

    if config.plot_all_predictions:
        predictions = []
        actual_values = []
        for x, y in test_data:
            predictions.append(trained_model.predict(x))
            actual_values.append(y)

        # Flatten lists
        predictions_flat = [item[0] for sublist in predictions for item in sublist]
        actual_flat = [float(item) for sublist in actual_values for item in sublist]

        # Get dates for validation set
        all_dates = list(test_dataset.index)
        dates = all_dates[config.sequence_length: (len(all_dates) - config.sequence_length) + 1]

        plt.plot(dates, predictions_flat)
        plt.plot(dates, actual_flat)
        plt.ylabel('Rainfall (mm)')
        plt.xlabel('date')
        plt.savefig("{}/plot.png".format(save_dir))


run_experiments()
