import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

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

# Model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    options=["Deeper Custom CNN (Recommended)", "Simpler CNN"],
    index=0  # Default to deeper model
)

# Load model based on choice (with caching for speed)
@st.cache_resource
def load_selected_model(choice):
    if choice == "Deeper Custom CNN (Recommended)":
        return load_model('best_model2_deeper.h5')  # Replace with your exact filename
    else:
        return load_model('best_model1_simpler.h5')  # Replace with your exact filename

model = load_selected_model(model_choice)

st.sidebar.markdown("### Model Info")
if model_choice == "Deeper Custom CNN (Recommended)":
    st.sidebar.success("Validation Accuracy: ~55%")
    st.sidebar.info("Higher capacity, better generalization")
else:
    st.sidebar.warning("Validation Accuracy: ~21%")
    st.sidebar.info("Lightweight baseline")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    img = image.resize((192, 256))  # Width 192, Height 256 → matches (256, 192) after transpose if needed
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

    # Results
    st.markdown("### Prediction Result")
    st.subheader(f"**{class_names[predicted_class].upper()}**")
    st.progress(confidence / 100)
    st.write(f"Confidence: **{confidence:.2f}%**")

    # All probabilities
    st.markdown("#### Confidence for All Classes")
    prob_df = {
        "Material": class_names,
        "Probability (%)": [f"{p:.2f}" for p in predictions[0] * 100]
    }
    st.table(prob_df)

    # Recyclability tip
    tips = {
        "cardboard": "♻️ Recycle as paper/cardboard.",
        "glass": "♻️ Recycle in glass bin.",
        "metal": "♻️ Recycle as metal (cans, foil).",
        "paper": "♻️ Recycle as paper.",
        "plastic": "♻️ Check local rules – many plastics are recyclable.",
        "trash": "🗑️ Non-recyclable – dispose in general waste."
    }
    st.info(tips[class_names[predicted_class]])

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Models trained from scratch on TrashNet-style dataset")