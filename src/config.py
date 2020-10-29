# Config file

process_data = False
load_model = True
plot_single_step_predictions = True
plot_all_predictions = True

""" The sizes of the hidden layers """
hidden_layers = [234]
learning_rate = 0.001
optimizer = 'adam'
loss = "mse"
train_split = 0.7
epochs = 25
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
selected_features = [0, 1, 2, 3, 5, 6, 7]
sequence_length = 7
