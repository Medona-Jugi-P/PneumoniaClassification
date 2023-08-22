from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

# Constants
IMAGE_SIZE = (256, 256)
NORMALIZATION_FACTOR = 255.0

app = Flask(__name__)

# Load the trained CNN model
model = tf.keras.models.load_model("pneumonia_trained_model.h5")

# Define the labels
labels = ['NORMAL', 'PNEUMONIA']

class Config:
    TEMPLATE_FOLDER = "templates"
    DEBUG = True

app.config.from_object(Config)

def process_uploaded_image(image_file):
    try:
        image = Image.open(image_file)
        return image
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return None

def preprocess_image(image):
    grayscale_image = image.convert('L')
    rgb_image = grayscale_image.convert('RGB')
    resized_image = rgb_image.resize(IMAGE_SIZE)
    image_array = np.array(resized_image)
    normalized_image = image_array / NORMALIZATION_FACTOR
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
    image_file = request.files.get('file')
    if not image_file:
        return "No image file uploaded."

    image = process_uploaded_image(image_file)
    if not image:
        return "Error processing image."

    predicted_label = predict_pneumonia(image)
    return render_template('result.html', label=predicted_label)

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])
