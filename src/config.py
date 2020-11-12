# Config file
experiments = True
load_model = False
plot_all_predictions = True
model_name = 'lstm_model'

""" The sizes of the hidden layers """
hidden_layer = 138
dense_layer = 138
learning_rate = 0.001
loss = 'mse'
epochs = 100
batch_size = 32
date_time_key = 'date'
sequence_length = 7
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

# Experiments configuration
models = {
    'gru': {
        'hidden_layer': 234,
        'dense_layer': 234,
        'learning_rate': 0.001
    },
    'lstm': {
        'hidden_layer': 234,
        'dense_layer': 170,
        'learning_rate': 0.001
    },
    'fully_connected': {
        'hidden_layer': 138,
        'dense_layer': 234,
        'learning_rate': 0.001
    },
    'simple_rnn': {
        'hidden_layer': 234,
        'dense_layer': 138,
        'learning_rate': 0.001
    }
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
