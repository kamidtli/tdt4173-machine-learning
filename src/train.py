# Training script
import keras

from src.models.gru import gru_model
from src.config import *

model = gru_model(hidden_layers=hidden_layer_sizes)

#Train
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss)()
"""
history = model.fit(
    dataset_train,
    epochs=epochs,
    validation_data=dataset_val,
    callbacks=[es_callback, modelckpt_callback],
)()
"""

