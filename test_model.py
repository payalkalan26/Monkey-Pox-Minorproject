# test_model.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Load the model
model_path = "models/monkeypox_mobilenetv3_final.h5"
model = load_model(model_path)

def test_image(img_path):
    try:
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)[0][0]
        confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)
        predicted_class = "Monkeypox" if prediction > 0.5 else "Non-monkeypox"
        
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"Raw prediction: {prediction:.4f}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence*100:.2f}%")
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")

# Test with sample images
test_images = [
    r"C:\Users\payal\CascadeProjects\monkeypox-demo\samples\monkeypox1.jpg",
    r"C:\Users\payal\CascadeProjects\monkeypox-demo\samples\normal1.jpg"
]

for img in test_images:
    if os.path.exists(img):
        test_image(img)
    else:
        print(f"File not found: {img}")