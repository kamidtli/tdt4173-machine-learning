from copy import deepcopy

import pandas as pd
import numpy as np


def fetch_raw_data() -> pd.DataFrame:
    data_2002_1 = pd.read_csv('../data/raw/data_2002_1.csv')
    data_2002_2 = pd.read_csv('../data/raw/data_2002_2.csv')
    data_2010_1 = pd.read_csv('../data/raw/data_2010_1.csv')
    data_2010_2 = pd.read_csv('../data/raw/data_2010_2.csv')

    # Merge attributes
    merged_1 = data_2002_1.merge(data_2002_2,
                                 indicator=True,
                                 how='inner',
                                 on=['Tid(norsk normaltid)'],
                                 suffixes=('', '_y'))

    merged_1.drop('_merge', 1, inplace=True)

    merged_1 = merged_1[[
        c for c in merged_1.columns if not c.endswith('_y')]]

    merged_2 = data_2010_1.merge(data_2010_2,
                                 indicator=True,
                                 how='inner',
                                 on=['Tid(norsk normaltid)'],
                                 suffixes=('', '_y'))

    merged_2.drop('_merge', 1, inplace=True)

    merged_2 = merged_2[[
        c for c in merged_2.columns if not c.endswith('_y')]]

    # Append datasets
    df = merged_1.append(merged_2)

    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:

    df.rename(columns={'Tid(norsk normaltid)': 'date', 'Navn': 'name', 'Stasjon': 'station', 'Midlere lufttrykk i stasjonsnivaa (dogn)': 'air_pressure', 'Vanndamptrykk (dogn)': 'water_vapor_pressure', 'Midlere relativ luftfuktighet (dogn)':
                       'relative_air_humidity', 'Spesifikk luftfuktighet': 'specific_air_humidity', 'Gjennomsnittlig skydekke (dogn)': 'average_cloud_cover', 'Middeltemperatur (dogn)': 'temperature', 'Kraftigste middelvind (dogn)': 'wind_speed', 'Nedbor (dogn)': 'downfall', 'Overskya vaer (dogn)': 'cloudy_weather'}, inplace=True)

    df.drop('name', 1, inplace=True)
    df.drop('station', 1, inplace=True)

    # Convert date from str to datetime
    df['date'] = pd.to_datetime(
        df['date'], format='%d.%m.%Y')

    df.sort_values('date', inplace=True)

    df.set_index('date', inplace=True)

    # Convert data types to float64
    columns = list(df)
    for x in range(len(columns)):
        df[columns[x]] = pd.to_numeric(df[columns[x]], errors='coerce')

    return df

def clean_string(input):
    string = deepcopy(input)
    string = string.lower()
    string = string.replace(" ", "_")
    return string

def preprocess_data(df: pd.DataFrame, N: int):
    dataset = pd.DataFrame()
    for i in range(N, len(df)):
        print(i)
        dataset.append({'x': df.iloc[i-N:i-1], 'y': df.iloc[i]})
    return dataset

def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std



