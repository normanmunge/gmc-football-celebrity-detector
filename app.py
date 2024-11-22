import os
import cv2
import numpy as np

'''
- Takes an input the image name and the desired output image size
- Loads the HAAR model to perform face detection task (the output size needs to be (160*160) pixels for the FaceNet model to work correctly
'''
def extract_face(image_path, required_size=(160, 160)):
    # Load the HAAR Cascade for face detection
    haar_model_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(haar_model_path)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None

    # Convert the image to grayscale for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print(f"No face detected in image: {image_path}")
        return None
    
    # Assuming the first detected face is the target
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    
    # Resize the face to the target size
    face_resized = cv2.resize(face, required_size)
    
    return face_resized


'''
- Takes a directory name as input to loop through its images
- Applies the ‘ extract face’ function. 
'''
def load_faces(dir_name, required_size=(160, 160)):
    faces = []
    for filename in os.listdir(dir_name):
        image_path = os.path.join(dir_name, filename)
        face = extract_face(image_path, required_size)
        if face is not None:
            faces.append(face)
    return faces

'''
- Takes the parent directory name and loops through its child directory to apply the load faces function
- Assign to them the corresponding label which is the child directory name.
'''
def load_dataset(parent_dir_name, required_size=(160, 160)):
    X, y = [], []
    for subdir in os.listdir(parent_dir_name):
        subdir_path = os.path.join(parent_dir_name, subdir)
        if not os.path.isdir(subdir_path):
            continue
        # Load all faces in the subdirectory
        faces = load_faces(subdir_path, required_size)
        # Assign the label (subdirectory name) to each face
        labels = [subdir] * len(faces)
        X.extend(faces)
        y.extend(labels)
    return np.array(X), np.array(y)

'''
Save the dataset
'''
def save_dataset(X, y, filename):
    np.savez_compressed(filename, X=X, y=y)
    print(f"Dataset saved to {filename}")


def main():
    keywords = ["Steven Gerrard","Mo Salah","Christiano Ronaldo","Messi","Wayne Rooney"]

    # List of folder names you want to create
    folders = [name.lower().replace(" ", "_") for name in keywords]

    combined_X, combined_y = [], []

    # Iterate over the list of folder names and create them
    for folder in folders:
        parent_dir = os.path.join("dataset/images", folder)
        X, y = load_dataset(parent_dir)
        print(f"Loaded {len(X)} faces with labels: {set(y)}")
        combined_X.extend(X)
        combined_y.extend(y)
    
    # Convert to numpy arrays
    combined_X = np.array(combined_X)
    combined_y = np.array(combined_y)

    if len(combined_X) > 0 and len(combined_y) > 0:
        output_file = "faces_dataset.npz"
        save_dataset(combined_X, combined_y, output_file)


if __name__ == '__main__':
    main()
    