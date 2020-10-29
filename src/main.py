import sys
from models.fully_connected import fully_connected_model
from models.lstm import lstm_model
from models.gru import gru_model
from train import train_model
from evaluate import model_loss, visualize_loss, show_plot
import config
from utils.csv_utils import to_csv, read_csv
from data.processor import process_data, fetch_raw_data, preprocess_data, normalize, clean_string, load_data
import pandas as pd
import tensorflow as tf
import numpy as np
import kerastuner as kt
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


# Set seed for reproducibility
"""
randomState = 14
np.random.seed(randomState)
tf.random.set_seed(randomState)
"""


def main():
    process_data = config.process_data
    load_model = config.load_model

    if process_data:
        raw_data = fetch_raw_data()
        df = process_data(raw_data)
        to_csv(df, '../data/processed/processed_data.csv')
    else:
        df = read_csv('../data/processed/processed_data.csv')

    df = df.fillna(0)
    selected = [config.features[i] for i in config.selected_features]
    print(
        "The selected features are:",
        ", ".join(selected),
    )
    index_downfall = selected.index("Downfall")

    dataset_train, dataset_val = load_data(df, selected, config, normalize_values=False)

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)
    print("Input: ", inputs.numpy()[0].shape)

    if load_model:
        model = tf.keras.models.load_model("../models/gru_model.h5")

    else:
        model = gru_model(
            hidden_layers=config.hidden_layers,
            shape=inputs,
            optimizer=config.optimizer,
            loss=config.loss,
            use_dropout=True
        )
        model, history = train_model(
            save_dir="../models",
            name="gru_model",
            model=model,
            epochs=config.epochs,
            dataset_train=dataset_train,
            dataset_val=dataset_val
        )
        print("Evaluate on test data")
        results = model.evaluate(dataset_val, batch_size=128)
        print("test loss, test acc:", results)
        visualize_loss(history, "Training and Validation loss")

    if config.plot_single_step_predictions:
        for x, y in dataset_val.take(5):
            show_plot(
                [x[0][:, index_downfall].numpy(), y[0].numpy(), model.predict(x)[0], np.average(x[0][:, index_downfall].numpy())],
                sequence_length=config.sequence_length,
                title="Single Step Prediction"
            )

    if config.plot_all_predictions:
        predictions = []
        actual_values = []
        for x, y in dataset_val:
            predictions.append(model.predict(x))
            actual_values.append(y)

        # Flatten lists
        predictions_flat = [item for sublist in predictions for item in sublist]
        actual_flat = [float(item) for sublist in actual_values for item in sublist]

        # Get dates for validation set
        dates = list(df.index[-len(actual_flat):])

        plt.plot(dates, predictions_flat)
        plt.plot(dates, actual_flat)
        plt.ylabel('Rainfall (mm)')
        plt.xlabel('date')
        plt.show()


if __name__ == "__main__":
    main()
