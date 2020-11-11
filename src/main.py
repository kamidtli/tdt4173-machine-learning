import matplotlib.pyplot as plt
import tensorflow as tf
from pandas.plotting import register_matplotlib_converters
import numpy as np

import config as config
from evaluate import visualize_loss
from models.fully_connected import fully_connected_model, fully_connected_model2
from models.gru import gru_model, gru_model2
from models.lstm import lstm_model2, lstm_model
from models.simple_rnn import simple_rnn_model2
from train import run_experiments, train_model
from utils.csv_utils import read_csv
from data.processor import load_data
from utils.plotting import plot_predictions

register_matplotlib_converters()


def main():
    # Set seed for reproducibility
    randomState = 14
    np.random.seed(randomState)
    tf.random.set_seed(randomState)

    model_name = config.model_name
    load_model = config.load_model
    experiments = config.experiments

    if experiments:
        run_experiments()

    else:
        train_data, train_dataset, val_data, val_dataset, test_data, test_dataset = load_data(config.features,
                                                                                              config.sequence_length)

        for batch in train_data.take(1):
            inputs, targets = batch
            break

        if load_model:
            trained_model = tf.keras.models.load_model("../models/{}.h5".format(model_name))
        else:
            save_dir = "../models/{}".format(model_name)
            model = lstm_model(
                hidden_layer=config.hidden_layer,
                dense_layer=config.dense_layer,
                shape=inputs,
                use_dropout=True
            )

            trained_model, history = train_model(
                save_dir=save_dir,
                name=model_name,
                model=model,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                dataset_train=train_data,
                dataset_val=val_data
            )
            print("Evaluate on test data")
            results = trained_model.evaluate(test_data, batch_size=32)
            print("test loss, test acc:", results)
            visualize_loss(history, "Training and Validation loss", save_dir=save_dir)

        if config.plot_all_predictions:
            plot_predictions(test_data, test_dataset, trained_model, config.sequence_length)


if __name__ == "__main__":
    main()
