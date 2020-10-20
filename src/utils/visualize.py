from copy import deepcopy

from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path

from csv_utils import read_csv

titles = [
    #"Air Pressure",
    "Water vapor pressure",
    "Relative air humidity",
    "Specific air humidity",
    "Average cloud cover",
    "Temperature",
    "Wind speed",
    "Downfall",
    "Cloudy weather",
]

def clean_string(input):
    string = deepcopy(input)
    string = string.lower()
    string = string.replace(" ", "_")
    return string


date_time_key = "date"
df = read_csv('../../data/processed/processed_data.csv')
x = df[[clean_string(titles[1])]]

def show_raw_visualization(data):
    Path("../../plots/data").mkdir(parents=True, exist_ok=True)
    for i in titles:
        title = clean_string(i)
        x = data[[title]]
        fig = x.plot().get_figure()
        fig.savefig("../../plots/data/{}.png".format(title))


show_raw_visualization(df)