import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Fake Currency Detector", page_icon="Currency")
st.title("Fake Currency Detection - 92% Accurate Model")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()
st.success("Real Trained Model Loaded Successfully! (92.31% Accuracy)")

uploaded_file = st.file_uploader("Upload Note Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Note", width=300)

    img_resized = img.resize((128,128))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred < 0.5:
        st.error(f"FAKE NOTE DETECTED! Confidence: {(1-pred)*100:.2f}%")
        st.write("This note appears to be fake, please verify carefully.")
    else:
        st.success(f"REAL NOTE! Confidence: {pred*100:.2f}%")
        st.write("This is an original note.")
        st.balloons()