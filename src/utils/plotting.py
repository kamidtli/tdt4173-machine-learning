from matplotlib import pyplot as plt


def plot_predictions(test_data, test_dataset, trained_model, sequence_length, save_dir='', save=False):

    # Plot predricted values
    plot_prediction(test_data, test_dataset, trained_model, sequence_length)

    # Plot actual values
    plot_actual(test_data, test_dataset, sequence_length)

    plt.ylabel('Rainfall (mm)')
    plt.xlabel('date')
    if save:
        plt.savefig("{}/plot.png".format(save_dir))
    else:
        plt.show()
    plt.clf()


def plot_multiple_models(model_data, save_dir='', save=False):

    for item in model_data:
        plot_prediction(item['test_data'], item['test_dataset'], item['model'], item['sequence_length'], item['features'], item['type'])

    plot_actual(model_data[0]['test_data'], model_data[0]['test_dataset'], model_data[0]['sequence_length'])

    plt.ylabel('Rainfall (mm)')
    plt.xlabel('date')
    plt.legend(loc="upper left")
    if save:
        plt.savefig("{}/plot.png".format(save_dir))
    else:
        plt.show()
    plt.clf()


def plot_prediction(test_data, test_dataset, trained_model, sequence_length, features=None, model_type=None):
    predictions = []
    for x, y in test_data:
        predictions.append(trained_model.predict(x))

    # Flatten list
    predictions_flat = [item[0] for sublist in predictions for item in sublist]

    # Get dates for validation set
    all_dates = list(test_dataset.index)
    dates = all_dates[sequence_length: (len(all_dates) - sequence_length) + 1]

    if features and model_type:
        plt.plot(dates, predictions_flat, label='{}-sequence-length-{}-features-{}'.format(model_type, sequence_length, features))
    else:
        plt.plot(dates, predictions_flat)


def plot_actual(test_data, test_dataset, sequence_length):
    actual_values = []
    for x, y in test_data:
        actual_values.append(y)

    # Flatten list
    actual_flat = [float(item) for sublist in actual_values for item in sublist]

    # Get dates for validation set
    all_dates = list(test_dataset.index)
    dates = all_dates[sequence_length: (len(all_dates) - sequence_length) + 1]

    plt.plot(dates, actual_flat, label='actual value')
