import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

# Load the trained model
model = tf.keras.models.load_model('Digit_Classifier.keras')

def preprocess_image(image_path):
    """
    Load and preprocess the image to match the model input.
    """
    try:
        img = Image.open(image_path).convert('L')  # convert to grayscale
        img = img.resize((28, 28))  # resize to 28x28
        img_array = np.array(img)
        img_array = 255 - img_array  # invert colors: MNIST has white digits on black
        img_array = img_array / 255.0  # normalize to [0,1]
        img_array = img_array.reshape(1, 28, 28)  # reshape for model
        return img_array
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

def predict_digit(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_digit, confidence

img = 'seven.jpeg'
preprocess_image(img)
digit, confidence = predict_digit(img)
print(f"Predicted Digit: {digit} (Confidence: {confidence:.2f})")
