import os
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered")
st.title("Chest X-Ray Classifier")
# Load trained model
@st.cache_resource
def load_trained_model():
    return load_model("model.h5",compile=False)

model = load_trained_model()
class_names = ['COVID19', 'NORMAL', 'PNEUMONIA']



st.write("Upload your own chest X-ray or select one of our sample images to get a prediction.")
sample_dir = "./sample_images"  # put 3 images here
sample_files = {
    "Sample Img 1": "one.jpg",
    "Sample Img 2": "two.jpg",
    "Sample Img 3": "three.jpg"
}

option = st.selectbox("Or choose a sample X-ray:", ["None"] + list(sample_files.keys()))
uploaded_file = st.file_uploader("Upload your chest X-ray (jpg/png)", type=["jpg", "png", "jpeg"])
image_to_predict = None
label = None
if uploaded_file is not None:
    image_to_predict = Image.open(uploaded_file).convert("RGB")
    label = "Your uploaded image"
elif option != "None":
    image_to_predict = Image.open(os.path.join(sample_dir, sample_files[option])).convert("RGB")
    label = option

# --- Prediction ---
if image_to_predict:
    st.image(image_to_predict.resize((300, 300)), caption=label)
    img = image_to_predict.resize((224, 224))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    predicted_label = class_names[pred_index]
    confidence = prediction[0][pred_index] * 100

    st.success(f"**Prediction:** {predicted_label} ({confidence:.1f}%)")
st.write("For More Sample Image Use Kaggle Dataset")
st.write("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")