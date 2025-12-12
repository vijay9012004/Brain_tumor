import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------
# Load your trained model
# ---------------------------
MODEL_PATH = "brain_detection_final.h5"   # Change to your model file
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224  # or your model input size


# ---------------------------
# Preprocess function
# ---------------------------
def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ---------------------------
# Prediction function
# ---------------------------
def predict(image):
    processed = preprocess_image(image)
    prob = model.predict(processed)[0][0]

    if prob >= 0.5:
        return "Tumor Detected", prob
    else:
        return "No Tumor", prob


# ---------------------------
# Streamlit Frontend UI
# ---------------------------
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to detect if a tumor is present.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', width=300)

    # predict button
    if st.button("Analyze Image"):
        result, confidence = predict(image)

        st.write("## 🔍 Prediction Result")
        st.success(f"**{result}**")
        st.write(f"**Confidence:** `{confidence * 100:.2f}%`")
