import matplotlib.pyplot as plt
import tensorflow as tf
from pandas.plotting import register_matplotlib_converters

from src.data.processor import process_dataset
import src.config as config
from src.evaluate import visualize_loss
from src.models.fully_connected import fully_connected_model
from train import train_model
from src.utils.csv_utils import read_csv

register_matplotlib_converters()


# Set seed for reproducibility

def main():
    model_name = config.model_name
    load_model = config.load_model

    train_dataset = read_csv('data/train_data.csv')
    train_data = process_dataset(
        train_dataset,
        features=config.features,
        flattened=True,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size
    )

    val_dataset = read_csv('data/validation_data.csv')
    val_data = process_dataset(
        val_dataset,
        features=config.features,
        flattened=True,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size
    )

    test_dataset = read_csv('data/test_data.csv')
    test_data = process_dataset(
        test_dataset,
        features=config.features,
        flattened=True,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size
    )

    for batch in train_data.take(1):
        inputs, targets = batch
        break

    if load_model:
        model = tf.keras.models.load_model("../models/{}.h5".format(model_name))
    else:
        model = fully_connected_model(
            hidden_layers=config.hidden_layers,
            shape=inputs,
            optimizer=config.optimizer,
            loss=config.loss,
            use_dropout=True
        )
        model, history = train_model(
            save_dir="models",
            name=model_name,
            model=model,
            epochs=config.epochs,
            dataset_train=train_data,
            dataset_val=val_data
        )
        print("Evaluate on test data")
        results = model.evaluate(test_data, batch_size=32)
        print("test loss, test acc:", results)
        visualize_loss(history, "Training and Validation loss")

    if config.plot_all_predictions:
        predictions = []
        actual_values = []
        for x, y in test_data:
            predictions.append(model.predict(x))
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
        plt.show()


if __name__ == "__main__":
    main()
