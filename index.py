import streamlit as st
import cv2
import numpy as np
from PIL import Image
import joblib
from image_face_detection.extractor import get_embedding_vgg16, load_vgg16_embedding_model

# Load the trained SVM model and the label encoder
svm_model = joblib.load("models/svm_face_classifier.joblib")
label_encoder = joblib.load("models/encoder.joblib")

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

# Streamlit App
html_spacer_temp = """
<div style = "margin-top: 50px"></div>
"""

html_title_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;">Football Players Face Recognition App </h2>
</div>
"""
st.markdown(html_title_temp, unsafe_allow_html = True)

html_body_temp = """
<div style = "text-align:center; margin-top: 10px; margin-bottom: 30px;">Predicting the name of the player in the uploaded image</div>
"""
st.markdown(html_body_temp, unsafe_allow_html = True)

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg', 'webp'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        if image is not None:
            col1, col2,col3 = st.columns(3, vertical_alignment="top")
            
            with col1:
                st.header("Uploaded Image")
                st.image(image, caption="Uploaded Image", use_container_width=True)    

            # Extract face
            st.write("Processing...")
            extracted_face = extract_face(np.array(image))

            if extracted_face is not None:
                with col2:
                    st.header("Detected Face")
                    st.image(extracted_face, caption="Detected Face", use_container_width=True)
                
                # Get embedding
                embedding = get_embedding_vgg16(embedding_model, extracted_face)

                # Predict using SVM
                prediction = svm_model.predict([embedding])
                predicted_label = label_encoder.inverse_transform(prediction)[0]

                with col3:
                    st.header("Prediction")
                    container = st.container(border=True, height=300)
                    html_prediction_temp = """
                    <div style = "margin-top: 100px"></div>
                    """
                    container.markdown(html_prediction_temp, unsafe_allow_html = True)
                    container.write(f"{predicted_label.capitalize().replace('_', ' ')}")
            else:
                st.write("No face detected in the uploaded image.")
    except Exception as e:
        st.markdown(html_spacer_temp, unsafe_allow_html = True)
        container = st.container(border=True, height=300)
        
        html_image_error_temp = """
        <div style = "text-align:center; vertical-align:middle; color:red; line-height:250px;">Cannot identify the uploaded image!</div>
        """
        container.markdown(html_image_error_temp, unsafe_allow_html = True)