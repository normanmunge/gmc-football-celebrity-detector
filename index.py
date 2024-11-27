import streamlit as st
import cv2
import numpy as np
from PIL import Image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.layers import Flatten
#from sklearn.preprocessing import LabelEncoder
import joblib
from image_face_detection.extractor import get_embedding_vgg16, load_vgg16_embedding_model

# Load the trained SVM model and the label encoder
svm_model = joblib.load("models/svm_face_classifier.joblib")
label_encoder = joblib.load("models/encoder.joblib")

# Load VGG16 embedding model
# def load_vgg16_embedding_model():
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#     x = base_model.output
#     x = Flatten()(x)  # Flatten the output into a vector
#     embedding_model = Model(inputs=base_model.input, outputs=x)
#     return embedding_model

embedding_model = load_vgg16_embedding_model()

# Function to extract face from an image
def extract_face(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    return face
 
# Function to extract face from an image


# Function to get embeddings
# def get_embedding(face):
#     face = face.astype('float32')
#     face = preprocess_input(face)
#     face = np.expand_dims(face, axis=0)
#     embedding = embedding_model.predict(face)
#     return embedding[0]

# Streamlit App
st.title("Face Recognition App")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Extract face
    st.write("Processing...")
    extracted_face = extract_face(np.array(image))

    if extracted_face is not None:
        st.image(extracted_face, caption="Detected Face", use_container_width=True)
        
        # Get embedding
        embedding = get_embedding_vgg16(embedding_model, extracted_face)

        # Predict using SVM
        prediction = svm_model.predict([embedding])
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display result
        st.write(f"Prediction: **{predicted_label}**")
    else:
        st.write("No face detected in the uploaded image.")