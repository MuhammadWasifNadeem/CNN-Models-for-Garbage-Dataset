import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Waste Material Classifier")
st.markdown("Upload an image of waste to classify it into: **cardboard, glass, metal, paper, plastic, or trash**.")

# Class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Expected model filenames (must match exactly what's in your repo)
DEEPER_MODEL_PATH = "best_model2_deeper.h5"
SIMPLER_MODEL_PATH = "best_model1_simpler.h5"

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    options=["Deeper Custom CNN (Recommended)", "Simpler CNN"],
    index=0
)

# Verify files exist
if not os.path.exists(DEEPER_MODEL_PATH):
    st.error(f"Model file not found: {DEEPER_MODEL_PATH} — Make sure it's uploaded to the repo!")
    st.stop()

if not os.path.exists(SIMPLER_MODEL_PATH):
    st.error(f"Model file not found: {SIMPLER_MODEL_PATH} — Make sure it's uploaded to the repo!")
    st.stop()

# Load model with caching
@st.cache_resource(show_spinner="Loading model... (first time only)")
def load_selected_model(choice):
    if choice == "Deeper Custom CNN (Recommended)":
        st.info("Loading Deeper Custom CNN (~55% validation accuracy)")
        return load_model(DEEPER_MODEL_PATH)
    else:
        st.info("Loading Simpler CNN (~21% validation accuracy)")
        return load_model(SIMPLER_MODEL_PATH)

model = load_selected_model(model_choice)

# Sidebar info
st.sidebar.markdown("### Model Info")
if model_choice == "Deeper Custom CNN (Recommended)":
    st.sidebar.success("Validation Accuracy: ~55%")
    st.sidebar.info("Selected for deployment")
else:
    st.sidebar.warning("Validation Accuracy: ~21%")
    st.sidebar.info("Baseline model")

# File uploader
uploaded_file = st.file_uploader("Choose a waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess: resize to training size (height=256, width=192)
    img = image.resize((192, 256))  # width first, then height for PIL
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

    # Display results
    st.markdown("### Prediction")
    st.subheader(f"**{class_names[predicted_idx].upper()}**")
    st.progress(confidence / 100)
    st.write(f"**Confidence: {confidence:.2f}%**")

    # Probability table
    st.markdown("#### All Class Probabilities")
    prob_data = {
        "Material": class_names,
        "Probability (%)": [f"{p*100:.2f}" for p in predictions[0]]
    }
    st.table(prob_data)

    # Recycling tip
    tips = {
        "cardboard": "♻️ Recycle as cardboard/paper.",
        "glass": "♻️ Recycle in glass container bin.",
        "metal": "♻️ Recycle as metal (cans, foil).",
        "paper": "♻️ Recycle as paper.",
        "plastic": "♻️ Check local guidelines — many plastics are recyclable.",
        "trash": "🗑️ General waste — not recyclable."
    }
    st.info(tips[class_names[predicted_idx]])

# Footer
st.markdown("---")
st.caption("Custom CNNs trained from scratch on waste classification dataset | Deployed with Streamlit")
