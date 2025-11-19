# test_model_v3.py
# ✅ Monkeypox Image Classification - Final Testing Script

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Load the trained model ---
model_path = "models/monkeypox_mobilenetv3_final.h5"
print(f"🧠 Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!\n")

# --- Class labels (update if needed) ---
class_labels = ['Monkeypox', 'Non-Monkeypox']

# --- Take image input ---
test_image_path = input("📸 Enter image path to test: ").strip('"')

# --- Load and preprocess image ---
img = image.load_img(test_image_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# --- Make prediction ---
pred = model.predict(img_array)
predicted_class = np.argmax(pred, axis=1)[0]
confidence = np.max(pred)

# --- Output ---
print("\n🧩 Prediction Result:")
print(f"✅ The image is classified as: **{class_labels[predicted_class]}**")
print(f"🔹 Confidence: {confidence * 100:.2f}%")
