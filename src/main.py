import pandas as pd
import tensorflow as tf
import numpy as np
import kerastuner as kt

from data.processor import process_data, fetch_raw_data, preprocess_data, normalize, clean_string, load_data, process_dataset
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
    train_dataset = read_csv('../data/train_data.csv')
    train_data = process_dataset(
      train_dataset, 
      features=config.features, 
      sequence_length=config.sequence_length,
      batch_size=config.batch_size
    )

    val_dataset = read_csv('../data/validation_data.csv')
    val_data = process_dataset(
      val_dataset, 
      features=config.features, 
      sequence_length=config.sequence_length,
      batch_size=config.batch_size
    )
    
    test_dataset = read_csv('../data/test_data.csv')
    test_data = process_dataset(
      test_dataset, 
      features=config.features, 
      sequence_length=config.sequence_length,
      batch_size=config.batch_size
    )

    for batch in train_data.take(1):
        inputs, targets = batch
        break

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
        dataset_train=train_data,
        dataset_val=val_data
    )
    print("Evaluate on test data")
    results = model.evaluate(test_data, batch_size=32)
    print("test loss, test acc:", results) 
    visualize_loss(history, "Training and Validation loss")


if __name__ == "__main__":
    main()
