import matplotlib.pyplot as plt

def model_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show();

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_plot(plot_data, sequence_length, title):
    labels = ["History", "True Future", "Model Prediction", "Average"]
    marker = [".-", "rx", "go", ".-"]
    plt.title(title)
    for i, val in enumerate(plot_data):
        if i == 0:
            plt.plot(plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(sequence_length, plot_data[i], marker[i], markersize=10, label=labels[i])
    plt.legend()
    plt.xlabel("Time-Step")
    plt.show()
    return




