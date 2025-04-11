# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load model
model = load_model("denoiser_model.h5")

st.title("🧼 Image Denoising with Deep Learning")

uploaded_file = st.file_uploader("Upload a noisy grayscale image (28x28)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Preprocess the uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.fit(image, (28, 28))  # Resize to 28x28
    img_array = np.asarray(image).astype('float32') / 255.0
    input_img = np.reshape(img_array, (1, 28, 28, 1))

    # Predict (denoise)
    output_img = model.predict(input_img)[0].reshape(28, 28)

    # Display original and denoised
    st.subheader("🔧 Noisy Input")
    st.image(image, width=150)

    st.subheader("✨ Denoised Output")
    st.image(output_img, width=150, clamp=True)

    st.success("Denoising complete!")
