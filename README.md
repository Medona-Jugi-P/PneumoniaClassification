# Summary: Pneumonia Detection from Chest X-Rays using Deep Learning and Flask

In this project, we have developed an application that detects and classifies pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) model. The application is built using Flask, a Python web framework, and integrates the trained deep learning model to provide real-time predictions for pneumonia cases.

## Technologies Used

- **Flask**: A lightweight Python web framework used to create the user interface and handle web requests.
- **TensorFlow and Keras**: Deep learning libraries used for building, training, and deploying the pneumonia detection CNN model.
- **PIL (Python Imaging Library)**: Used for image processing tasks such as loading, resizing, and converting images.
- **NumPy**: Utilized for numerical operations on arrays, essential for image preprocessing and handling.
- **Matplotlib**: Employed for generating visualizations and plots.

## Approach

The project follows these steps:

1. **Load the Trained CNN Model**: The project loads a pre-trained CNN model for pneumonia detection.
2. **Set up a Flask Web Application**: A Flask web application is set up to handle user interactions and requests.
3. **User Image Upload**: Users can upload chest X-ray images through the web interface.
4. **Image Preprocessing and Prediction**: The uploaded image is preprocessed and passed through the CNN model for prediction.
5. **Display Predictions**: The application displays the prediction result on the web interface.

## Deployment

The project is deployed and accessible at: [Pneumonia Classification Web App](https://pneumoniaclassification.onrender.com)

## Conclusion

The developed application provides an accessible and user-friendly way for medical professionals and users to quickly assess chest X-ray images for pneumonia. By leveraging deep learning techniques and web technologies, this project demonstrates the potential of AI-driven tools in medical diagnosis. The application can aid in identifying pneumonia cases earlier, potentially leading to faster medical interventions and improved patient outcomes.
