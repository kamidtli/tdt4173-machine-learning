from copy import deepcopy

from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from data.processor import clean_string
from csv_utils import read_csv

titles = [
    "Air Pressure",
    "Water vapor pressure",
    "Relative air humidity",
    "Specific air humidity",
    "Average cloud cover",
    "Temperature",
    "Wind speed",
    "Downfall",
    "Cloudy weather",
]




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

def show_heatmap(data):
    plt.matshow(data.corr())
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.savefig("../../plots/{}.png".format("heatmap_correlation"))

#show_heatmap(df)

#show_raw_visualization(df)