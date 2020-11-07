import matplotlib.pyplot as plt
import tensorflow as tf
from pandas.plotting import register_matplotlib_converters

import config as config
from evaluate import visualize_loss
from models.fully_connected import fully_connected_model
from models.gru import gru_model
from train import train_model, run_experiments
from utils.csv_utils import read_csv
from utils.plotting import plot_predictions
from utils.load_data import load_data

register_matplotlib_converters()


def main():

    model_name = config.model_name
    load_model = config.load_model
    experiments = config.experiments

    if experiments:
        run_experiments()

    else:
        train_data, train_dataset, val_data, val_dataset, test_data, test_dataset = load_data(config.features, False, config.sequence_length)

        for batch in train_data.take(1):
            inputs, targets = batch
            break

        if load_model:
            model = tf.keras.models.load_model("../models/{}.h5".format(model_name))
        else:
            model = gru_model(
                hidden_layers=config.hidden_layers,
                shape=inputs,
                learning_rate=config.learning_rate,
                loss=config.loss,
                use_dropout=True
            )
            trained_model, history = train_model(
                save_dir="models",
                name=model_name,
                model=model,
                epochs=config.epochs,
                dataset_train=train_data,
                dataset_val=val_data
            )
            print("Evaluate on test data")
            results = trained_model.evaluate(test_data, batch_size=32)
            print("test loss, test acc:", results)
            #visualize_loss(history, "Training and Validation loss")

        if config.plot_all_predictions:
            plot_predictions(test_data, test_dataset, trained_model, config.sequence_length)


if __name__ == "__main__":
    main()
