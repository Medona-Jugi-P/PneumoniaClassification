# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 19:09:53 2023
@author: CC
"""

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__, template_folder="templates")

# Load the trained CNN model
model = tf.keras.models.load_model("pneumonia_trained_model.h5")

# Define the labels
labels = ['NORMAL', 'PNEUMONIA']


def preprocess_image(image):
    # Convert the image to grayscale
    grayscale_image = image.convert('L')

    # Convert the grayscale image to RGB
    rgb_image = grayscale_image.convert('RGB')

    # Resize the image to (256, 256)
    resized_image = rgb_image.resize((256, 256))

    # Convert the image to a numpy array
    image_array = np.array(resized_image)

    # Normalize the image array
    normalized_image = image_array / 255.0

    # Add an extra dimension to match the expected shape
    preprocessed_image = np.expand_dims(normalized_image, axis=0)

    return preprocessed_image


def predict_pneumonia(image):
    preprocessed_image = preprocess_image(image)
    result = model.predict(preprocessed_image)
    predicted_class = np.argmax(result)
    predicted_label = labels[predicted_class]
    return predicted_label


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    image_file = request.files['file']
    if image_file:
        # Read the image using PIL
        image = Image.open(image_file)

        # Perform prediction
        predicted_label = predict_pneumonia(image)

        # Render the prediction result
        return render_template('result.html', label=predicted_label)
    else:
        return "No image file uploaded."


if __name__ == '__main__':
    app.run(debug=True)
