import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
@st.cache_resource
def load_my_model():
    model = load_model('flower_classifier_mobilenetv2.h5')
    return model

model = load_my_model()

# Define class names (ensure these match your training data)
class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

IMG_SIZE = 224

st.title("Flower Image Classification")
st.write("Upload an image of a flower and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    img_array = tf.cast(img_array, tf.float32) / 255.0
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0])

    st.write(f"Prediction: {predicted_class_name}")
    st.write(f"Confidence: {confidence:.2f}")

    st.subheader("Top 3 Probabilities")
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    for i in top_3_indices:
        st.write(f"{class_names[i]}: {predictions[0][i]:.2f}")

st.subheader("Confusion Matrix")
st.image("confusion_matrix.png", caption="Model Evaluation on Test Dataset", use_column_width=True)



