import pandas as pd
import tensorflow as tf

from data.processor import process_data, fetch_raw_data, preprocess_data, normalize, clean_string
from utils.csv_utils import to_csv, read_csv
import config
from evaluate import model_loss
from train import train_model
from models.gru import gru_model
from models.fully_connected import fully_connected_model


def main():
    process_data_bool = True
    if process_data_bool:
        raw_data = fetch_raw_data()
        df = process_data(raw_data)
        to_csv(df, '../data/processed/processed_data.csv')
    else:
        df = read_csv('../data/processed/processed_data.csv')

    df = df.fillna(-1)
    selected = [config.features[i] for i in config.selected_features]
    print(
        "The selected features are:",
        ", ".join(selected),
    )
    selected = [clean_string(i) for i in selected]
    train_split = int(config.train_split * int(df.shape[0]))
    features = df[selected]
    features = normalize(features.values, train_split)
    features = pd.DataFrame(features)

    train_data = features.loc[0: train_split - 1]
    val_data = features.loc[train_split:]

    x_train = train_data[config.selected_features].values
    y_train = features.loc[1:train_split][[7]]  # 7 is the index of downfall

    x_val = val_data[config.selected_features].values
    y_val = features.loc[train_split:][[7]]

    dataset_train = tf.keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
    )
    dataset_val = tf.keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
    )
    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Target shape:", targets.numpy().shape)

    model = fully_connected_model(
        hidden_layers=config.hidden_layers,
        shape=inputs,
        optimizer=config.optimizer,
        loss=config.loss
    )
    history = train_model(
        save_dir="../models",
        name="gru_model",
        model=model,
        epochs=config.epochs,
        dataset_train=dataset_train,
        dataset_val=dataset_val
    )
    model_loss(history)

if __name__ == "__main__":
    main()
