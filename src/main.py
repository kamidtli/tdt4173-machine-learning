import pandas as pd
import tensorflow as tf
import numpy as np
import kerastuner as kt

from data.processor import process_data, fetch_raw_data, preprocess_data, normalize, clean_string, load_data
from utils.csv_utils import to_csv, read_csv
import config
from evaluate import model_loss, visualize_loss, show_plot
from train import train_model
from models.gru import gru_model
from models.lstm import lstm_model
from models.fully_connected import fully_connected_model

# Set seed for reproducibility
randomState = 14
np.random.seed(randomState)
tf.random.set_seed(randomState)

def main():
    process_data_bool = True
    if process_data_bool:
        raw_data = fetch_raw_data()
        df = process_data(raw_data)
        to_csv(df, '../data/processed/processed_data.csv')
    else:
        df = read_csv('../data/processed/processed_data.csv')

    df = df.fillna(0)
    index_downfall = df.columns.get_loc("downfall")
    print("index of downfall: ", index_downfall)
    selected = [config.features[i] for i in config.selected_features]
    print(
        "The selected features are:",
        ", ".join(selected),
    )

    dataset_train, dataset_val = load_data(df, selected, config, normalize_values=False)

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)
    print("Input: ", inputs.numpy()[0].shape)

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
    for x, y in dataset_val.take(5):
        print(x[0])
        print(x[0][:,index_downfall].numpy())
        show_plot(
            [x[0][:, index_downfall].numpy(), y[0].numpy(), model.predict(x)[0]],
            title="Single Step Prediction"
        )

if __name__ == "__main__":
    main()
