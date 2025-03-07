from tensorflow.keras.models import load_model
import os

def load_trained_model():
    """
    Loads and returns the trained model from the absolute path.
    """
    model_path = os.path.abspath("src/model.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model not found at {model_path}")

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        raise IOError(f"Error loading the model: {e}")

def predict_image(model, image):
    """
    Receives the model and a preprocessed image, performing the prediction.
    """
    import numpy as np
    image = image.reshape((1, *image.shape))
    prediction = model.predict(image)
    return prediction
