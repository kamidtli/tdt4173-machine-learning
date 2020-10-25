# Training script
import tensorflow as tf
from pathlib import Path

def train_model(save_dir, name, model, epochs, dataset_train, dataset_val):
    # Train
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    history = model.fit(
        dataset_train,
        epochs=epochs,
        validation_data=dataset_val,
        callbacks=[tensorboard_callback],
    )
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save("{}/{}.h5".format(save_dir, name))
    return history

