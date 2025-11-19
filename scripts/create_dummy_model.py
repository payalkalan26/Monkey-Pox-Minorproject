import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "monkeypox_cnn.h5")

os.makedirs(MODELS_DIR, exist_ok=True)

# Simple CNN for 224x224x3 -> 2 classes
inputs = keras.Input(shape=(224, 224, 3))
x = layers.Conv2D(16, 3, activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(2, activation="softmax")(x)
model = keras.Model(inputs, outputs, name="dummy_monkeypox_cnn")

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Save untrained model (structure + random weights) to H5
model.save(MODEL_PATH)
print(f"Saved dummy model to: {MODEL_PATH}")
