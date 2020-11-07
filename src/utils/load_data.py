import config
from data.processor import process_dataset

from utils.csv_utils import read_csv


def load_data(features, flattend, sequence_length):
    train_dataset = read_csv('../data/train_data.csv')
    train_data = process_dataset(
        train_dataset,
        features=features,
        flattened=flattend,
        sequence_length=sequence_length,
        batch_size=config.batch_size
    )

    val_dataset = read_csv('../data/validation_data.csv')
    val_data = process_dataset(
        val_dataset,
        features=features,
        flattened=flattend,
        sequence_length=sequence_length,
        batch_size=config.batch_size
    )

    test_dataset = read_csv('../data/test_data.csv')
    test_data = process_dataset(
        test_dataset,
        features=features,
        flattened=flattend,
        sequence_length=sequence_length,
        batch_size=config.batch_size
    )

    return train_data, train_dataset, val_data, val_dataset, test_data, test_dataset
