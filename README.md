# Football Player Celebrity Detector

The purpose of this project is to build a Face Recognition System using FaceNet/VGG16 on football players. The players are Steven Gerrard, Mo Salah, Christiano Ronaldo, Wayne Rooney and Messi.

For getting images to train the system, I scrapped Google Images and downloaded each of their faces.

## 1. Folder Structure

### 1.1. Folder Structure

The repository contains:

1. [Datasets Directory](/datasets/) - This directory holds our dataset - images split into train and test directories.

2. [Requirements.txt] (/requirements.txt) - Showing the necessary libraries and dependencies to run the notebook.
3. Files
   - [app.py](app.py) - A python script that loads image from a dataset, extracts faces from them and saves into a .npz file
   - [web-scraping-main.py](web-scraping-main.py) - A python script to scrap data off Google Images based on search query
   - [image_downloader.py](image_downloader.py) - A python script to download images and save them into our dataset directory
   - [faces_dataset.npz](faces_dataset.npz) - A .npz file that contains the extracted faces.
   - [haarcascade_frontalface_default.xml] - Viola Davis Algorithm to detect faces from an image/video.
4. [Gitignore File](/.gitignore) - File to ignore files and directories from being pushed to the remote repository
5. [README.md File](/README.md) - Guiding instructions for describing and running the project.
6. [excel_links](excel_links) -  Directory where we save the image urls to scrapped off the internet for our football celebrities saved in Excel format.

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

The above listed libraries are the core ones used in the repository. However, during installation you'll notice other dependencies installed that enable to work as expected. They are highlighted on the [requirement.txt](/requirements.txt) file.

### 3.2. Installation of the Libraries

Ensure you are have a version python > 3 installed and running on your local environment in order to to be able to install the libraries and run the notebook. Afterwards, ensure the virtual environment you created above is active before running the installation commands below on your terminal.

To install the libraries run:

```bash
pip install pandas numpy matplotlib seaborn jupterlab chardet scikit-learn
```

You can also install all the libraries by running:

```bash
pip install requirements.txt
```

## 4. Running your script

To run your script in your local environment, run the command below on your terminal starting with the web-scraping-main.py, then image_downloader.py and finally app.py:

```bash
python (name_of_file).py
```
