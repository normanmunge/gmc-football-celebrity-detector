'''
INSTRUCTIONS
-------------
1. Choose 1 to 5 people to collect their images, try to collect 40 per person : 30 images for the train set and 10 for the test set.
2. Store the collected images following a predefined structure given by the instructor
3. Extract faces from the collected images and assign them to the appropriate labels. Exciting face detection models are available to use such as the HAAR cascade model.
4. Run through the commands below
5. Execute the previous routine on the train set as well as on the test set
6. Save the generated images for both sets into the same compressed ‘.npz’ file.
'''



'''
- Takes an input the image name and the desired output image size
- Loads the HAAR model to perform face detection task (the output size needs to be (160*160) pixels for the FaceNet model to work correctly
'''
def extract_face(image, required_size=(160, 160)):
    pass


'''
- Takes a directory name as input to loop through its images
- Applies the ‘ extract face’ function. 
'''
def load_faces(dir_name):
    pass

'''
- Takes the parent directory name and loops through its child directory to apply the load faces function
- Assign to them the corresponding label which is the child directory name.
'''
def load_dataset(dir_name):
    pass


def main():
    pass


if __name__ == '__main__':
    main()
    