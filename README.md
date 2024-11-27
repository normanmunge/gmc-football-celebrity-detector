# Football Player Celebrity Detector

The purpose of this project is to build a Face Recognition System using FaceNet/VGG16 on football players. The players are Steven Gerrard, Mo Salah, Christiano Ronaldo, Wayne Rooney and Messi.

For getting images to train the system, I scrapped Google Images and downloaded each of their faces.

## 1. Folder Structure

### 1.1. Folder Structure

The repository contains:

1. [Datasets Directory](/datasets/) - This directory holds our dataset - images split into train and test directories.

2. [Requirements.txt] (/requirements.txt) - Showing the necessary libraries and dependencies to run the notebook.
3. Files
   - [web-scraping-main.py](image_extractor_scripts/web-scraping-main.py) - A python script to scrap data off Google Images based on search query
   - [image_downloader.py](image_extractor_scripts/image_downloader.py) - A python script to download images and save them into our dataset directory
   - [app.py](image_face_detection/app.py) - A python script that loads image from a dataset, extracts faces from them and saves into a .npz file
   - [extractor.py](image_face_detection/extractor.py) - A python script that saves the embeddings of the extracted faces.
   - [faces_dataset.npz](faces_dataset.npz) - A .npz file that contains the extracted faces.
   - [vgg16_face_embeddings.npz](vgg16_face_embeddings.npz) - A .npz file that contains the embeddings of the extracted faces.
   - [haarcascade_frontalface_default.xml](/haarcascade_frontalface_default.xml) - Viola Davis Algorithm to detect faces from an image/video.
   - [index.py](/index.py) - Python script to spin up a Streamlit Web App
4. [Gitignore File](/.gitignore) - File to ignore files and directories from being pushed to the remote repository
5. [README.md File](/README.md) - Guiding instructions for describing and running the project.
6. [excel_links](excel_links) - Directory where we save the image urls to scrapped off the internet for our football celebrities saved in Excel format.

## 2. Setting up your local environment

This section guides you on how to setup your environment and run the repository on your local environment.

### 2.1. Creating a virtual environment

Create a virtual environment to install and manage the libraries to isolate them from your global environments.

To create a virtual environment, run the command below on your terminal:

```bash
python -m venv 'myenv_name'
```

Disclaimer: This approach is recommended for Linux and Mac environments. There might be a different approach for setting up in Windows environments.

### 2.2. Activating your environment

To activate your environment on linux or mac operating system. Run the command below on your terminal.

```bash
source /path/to/myenv_name/bin/activate
```

To activate your environment on a windows environment:

```bash
source \path\to\myenv_name\Scripts\activate
```

### 2.3. Deactivating your environment

Once you're done working on the repository, <b><i>REMEMBER</i></b> to deactivate your virtual environment in order to separate your local project dependencies.

To deactivate your environment on a linux or mac operating system. Run the command below on your terminal:

```bash
deactivate
```

## 3. Libraries and Installations

### 3.1. Required Libraries

The important libraries used in this environment are:

1. Pandas - Used for manipulation, exploration, cleaning and analyzing of your dataset.
2. Numpy - Used for mathematical and statistical purposes often to prepare your dataset for machine learning
3. OpenCV Python - Used to manipulate images in python
4. Selenium - Used to test web browsers and scrap off information
5. Keras
6. Keras-facenet
7. Tensorflow

The above listed libraries are the core ones used in the repository. However, during installation you'll notice other dependencies installed that enable to work as expected. They are highlighted on the [requirement.txt](/requirements.txt) file.

### 3.2. Installation of the Libraries

Ensure you are have a version python > 3 installed and running on your local environment in order to to be able to install the libraries and run the notebook. Afterwards, ensure the virtual environment you created above is active before running the installation commands below on your terminal.

To install the libraries run:

```bash
pip install -r requirements.txt
```

You can also install all the libraries by running:

```bash
pip install requirements.txt
```

## 4. Running your script

The steps below will help you to run the project from scrapping the web for the football celebrity images to running it on a streamlit app.

#### 1. Scrapping the web and downloading the photos

This phase scraps google images using Selenium in a Chrome web driver and saves the image urls of the football celebrites in an excel file in the [excel_links directory](/excel_links).

- Run the [web-scrapping-main.py](/image_extractor_scripts/web-scraping-main.py) script

```bash
python image_extractor_scripts/web-scraping-main.py
```

Thereafter, we read the excel files using Pandas and loop through each of them to download the images. These images are downloaded in the [dataset/images directory](/dataset/images/).

- Run the [image_downloader.py](/image_extractor_scripts/image_downloader.py) script

```bash
python image_extractor_scripts/image_downloader.py
```

#### 2. Run the Viola Davis model to detect faces, extract the faces and save their embeddings

Here, we use the OpenCV library and the [viola davis model](/haarcascade_frontalface_default.xml) to detect faces from your downloaded images which are saved in [faces_dataset.npz](/faces_dataset.npz) compressed file.

- Run the [app.py](/image_face_detection/app.py) script

```bash
python image_face_detection/app.py
```

Thereafter use the vgg16 model to extract the embeddings for each of the images and save them in a [embeddings.npz](/vgg16_face_embeddings.npz) compressed file.

- Run the [extractor.py](/image_face_detection/extractor.py) script.

```bash
python image_face_detection/extractor.py
```

#### 3. Training a classifier model

The third step entails training and evaluating a classifier model with the image embeddings. In this case we're using the SVM and the RandomForestClassifier to train the model. We afterwards save the model in a joblib format. This is done on this [notebook](/image_classification_model/svm_image_classifier.ipynb)

Open the notebook using jupyter lab by running the command below on your terminal:

```bash
jupyter lab
```

Once the notebook is open, train the model. It will save an [enocder.joblib](/models/encoder.joblib) needed to convert data in machine readable format and a [model.joblib](/models/svm_face_classifier.joblib) to save the model that will be used in our web app to detect the celebrity based on a user's uploaded image.

#### 4. Running a Streamlit App

For our model to detect the celebrity, we spin up a web application where users can upload a face and the app returns a prediction based on our classifier model detecting the name of the celebrity.

In your terminal, run:

```bash
streamlit run index.py
```

This spins up the app locally in your browser

#### Note

To run any of the python scripts in your local environment, run the command below on your terminal:

```bash
python (path_to_file).py
```
