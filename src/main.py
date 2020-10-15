from data.processor import process_data, fetch_raw_data
from utils.csv import to_csv, read_csv


def main():
    process_data_bool = True
    if process_data_bool:
        raw_data = fetch_raw_data()
        df = process_data(raw_data)
        to_csv(df, '../data/processed/processed_data.csv')
    else:
        df = read_csv('../data/processed/processed_data.csv')

    print(df)


if __name__ == "__main__":
    main()
