# Config file
""" The sizes of the hidden layers """
hidden_layers = [128, 64]
learning_rate = 0.001
optimizer = 'adam'
loss = "mse"
train_split = 0.7
epochs = 10
batch_size = 32
date_time_key = "date"
features = [
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
selected_features = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sequence_length = len(selected_features)
