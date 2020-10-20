from data.processor import process_data, fetch_raw_data
from utils.csv import to_csv, read_csv

from config import train_split
from evaluate import model_loss
from train import train_model


def main():
    process_data_bool = False
    if process_data_bool:
        raw_data = fetch_raw_data()
        df = process_data(raw_data)
        to_csv(df, '../data/processed/processed_data.csv')
    else:
        df = read_csv('../data/processed/processed_data.csv')

    print(df[:1])

    split = int(train_split * int(df.shape[0]))
    #train_data = df.loc[0 : split - 1]
    #val_data = df.loc[split:]

    #history = train_model()
    #model_loss(history)

if __name__ == "__main__":
    main()
