import streamlit as st
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import numpy as np
from PIL import Image
import os

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Deep Learning Model Prediction App", layout="centered")
st.title("Deep Learning Model Prediction App")

# -------------------------
# Define file paths (all in project root)
# -------------------------
H5_MODEL_FILE = "best_model.h5"
KERAS_MODEL_FILE = "model.keras"
PICKLE_FILE = "model.pkl"
CONFIG_FILE = "model_config.json"

# -------------------------
# Load config
# -------------------------
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
else:
    st.warning("model_config.json not found. Using default values.")
    config = {
        "input_shape": [224, 224, 3],
        "class_names": ["Class A", "Class B", "Class C"]
    }

input_shape = tuple(config.get("input_shape", [224, 224, 3]))
class_names = config.get("class_names", ["Class A", "Class B", "Class C"])

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    try:
        if os.path.exists(H5_MODEL_FILE):
            model = keras.models.load_model(H5_MODEL_FILE)
        elif os.path.exists(KERAS_MODEL_FILE):
            model = keras.models.load_model(KERAS_MODEL_FILE)
        else:
            st.error("No model file found!")
            return None
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -------------------------
# Load preprocessing objects
# -------------------------
@st.cache_resource
def load_preprocessing():
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as f:
            preprocessing = pickle.load(f)
        return preprocessing
    else:
        st.warning("No preprocessing.pkl found, skipping preprocessing.")
        return None

preprocessing = load_preprocessing()

# -------------------------
# Helper function: preprocess image
# -------------------------
def preprocess_input(image: Image.Image):
    image = image.resize((input_shape[0], input_shape[1]))
    img_array = np.array(image)
    img_array = img_array / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------
# User input
# -------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_input(image)

    if model is not None:
        prediction = model.predict(input_data)[0][0]  # scalar value

        if prediction >= 0.5:
          predicted_label = "REAL"
          confidence = prediction
        else:
          predicted_label = "FAKE"
          confidence = 1 - prediction

        st.success(f"Predicted Class: {predicted_label}")
        st.info(f"Confidence: {(confidence*100):.2f}")
    else:
        st.error("Model could not be loaded. Prediction skipped.")