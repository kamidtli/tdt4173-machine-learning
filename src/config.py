
# Config file
experiments = True
load_model = False
plot_all_predictions = True
model_name = 'gru_model'

""" The sizes of the hidden layers """
hidden_layers = [234, 10]
learning_rate = 0.001
loss = 'mse'
train_split = 0.7
epochs = 100
batch_size = 32
date_time_key = 'date'
features = [
    'Air Pressure',
    'Water vapor pressure',
    'Relative air humidity',
    'Specific air humidity',
    # 'Average cloud cover',
    'Temperature',
    'Wind speed',
    'Downfall',
    # 'Cloudy weather',
]
selected_features = [0, 1, 2, 3, 5, 6, 7]
sequence_length = 1

# Experiments configuration
models = {
    'gru': {
        'hidden_layers': [234, 10],
        'learning_rate': 0.001
    },
    'lstm': {
        'hidden_layers': [202, 170, 170],
        'learning_rate': 0.01
    },
    'fully_connected': {
        'hidden_layers': [170, 10, 10],
        'learning_rate': 0.01
    },
    # 'simple_rnn': {
    #     'hidden_layers': [234, 138, 10],
    #     'learning_rate': 0.01
    # }
}

sequence_lengths = [1, 3, 7, 14]
experiment_features = {
    '4': [
        'Air Pressure',
        'Specific air humidity',
        'Wind speed',
        'Downfall',
    ],
    '7': [
        'Air Pressure',
        'Water vapor pressure',
        'Relative air humidity',
        'Specific air humidity',
        'Temperature',
        'Wind speed',
        'Downfall',
    ],
    '9': [
        'Air Pressure',
        'Water vapor pressure',
        'Relative air humidity',
        'Specific air humidity',
        'Average cloud cover',
        'Temperature',
        'Wind speed',
        'Downfall',
        'Cloudy weather',
    ]
}
