from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Flatten
import numpy as np

"""
Load the pre-trained VGG16 model and modify it to output embeddings.
Returns:
    A modified VGG16 model for face embeddings.
"""
# Load pre-trained VGG16 model without the top classification layer
def load_vgg16_embedding_model():
    # Load the VGG16 model pre-trained on ImageNet, exclude the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
    
    # Add a custom output layer for embeddings
    x = base_model.output
    x = Flatten()(x)  # Flatten the output into a vector
    embedding_model = Model(inputs=base_model.input, outputs=x)
    
    return embedding_model

"""
Calculate the embedding for a given face using the pre-trained VGG16 model.
Args:
    model: The embedding model (VGG16).
    face: The face image as a numpy array.
Returns:
    The embedding vector for the face.
"""
# Get the embedding for a face image using VGG16
def get_embedding_vgg16(model, face):
    # Preprocess the image for VGG16
    face = face.astype('float32')
    face = preprocess_input(face)  # Normalizes for VGG16
    
    # Add batch dimension
    face = np.expand_dims(face, axis=0)
    
    # Get the embedding
    embedding = model.predict(face)
    return embedding[0]

"""
Process all face images to calculate their embeddings and save to an .npz file.
Args:
    model: The embedding model VGG16.
    face_images: A numpy array of face images.
    labels: A numpy array of labels corresponding to the images.
    output_file: The name of the output .npz file to save the embeddings.
"""
# Function to process all faces and calculate embeddings
def process_and_save_embeddings(model, face_images, labels, output_file):
    embeddings = []
    for face in face_images:
        embedding = get_embedding_vgg16(model, face)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    np.savez_compressed(output_file, embeddings=embeddings, labels=labels)
    print(f"Embeddings saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Load the VGG16 model for embeddings
    vgg16_model = load_vgg16_embedding_model()
    
    # Load dataset
    data = np.load("faces_dataset.npz")
    face_images, labels = data['X'], data['y']
    
    # Process and save embeddings
    output_embedding_file = "vgg16_face_embeddings.npz"
    process_and_save_embeddings(vgg16_model, face_images, labels, output_embedding_file)