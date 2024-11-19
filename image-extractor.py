import cv2
import streamlit as st
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #a pre-trained model that can be used to detect faces in images and videos


def hex_to_rbg(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0,2, 4))
    
def detect_faces(image):
    # Read the input image
    img = cv2.imread(image)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output image
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def save_image(img):
    cv2.imwrite('images/face_detection.png', img)
    
def upload_image():
    #images = []
    uploaded_files = st.file_uploader(
        "Choose a picture", accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            image_name = uploaded_file.name
            st.write("filename:", image_name)
            #st.write(bytes_data)
            
            detect_faces(image_name)
            
            
    
    
    
def app():
    html_title_temp = """
    <div style="background:#025246 ;padding:10px; margin-bottom:30px">
    <h2 style="color:white;text-align:center;">Face Detection App </h2>
    </div>
    """
    st.markdown(html_title_temp, unsafe_allow_html = True)

    upload_image()
        
        # if len(images):
        #     print('GETTING HERE')
        #     st.write("filename:", images[0].name)
        #     st.write(images[0].bytes_data)
    
    
    #st.sidebar.header("User Parameters")
    #hex_color = st.sidebar.color_picker("Pick a rectangle color", "#00FF00")
    #neighbors = st.sidebar.slider("Min Neighbors i.e how many neighbors each rectangle should have to retain it.", 1, 10, 5)
    #scale = st.sidebar.slider("Scale Factor i.e how much size of the face do you want to be detected?", 1.1, 2.0, 1.3, step=0.1)

    # if st.button("Start Detection"):
    #     detect_faces(hex_color, neighbors, scale) "#00FF00"
        
if __name__ == '__main__':
    app()