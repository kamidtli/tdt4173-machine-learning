from matplotlib import pyplot as plt


def plot_predictions(test_data, test_dataset, trained_model, sequence_length, save_dir='', save=False):
    predictions = []
    actual_values = []
    for x, y in test_data:
        predictions.append(trained_model.predict(x))
        actual_values.append(y)

    # Flatten lists
    predictions_flat = [item[0] for sublist in predictions for item in sublist]
    actual_flat = [float(item) for sublist in actual_values for item in sublist]

    # Get dates for validation set
    all_dates = list(test_dataset.index)
    dates = all_dates[sequence_length: (len(all_dates) - sequence_length) + 1]

    plt.plot(dates, predictions_flat)
    plt.plot(dates, actual_flat)
    plt.ylabel('Rainfall (mm)')
    plt.xlabel('date')
    if save:
        plt.savefig("{}/plot.png".format(save_dir))
    else:
        plt.show()
    plt.clf()
