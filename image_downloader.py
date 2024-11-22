import pandas as pd
import requests

# creating the function that saves images
def download_images(url,count,query):

    try:
    
        img_data = requests.get(url).content
        with open(f'dataset/images/{query}/{query}_{count}.jpg', 'wb') as handler:
            handler.write(img_data)
    
    
    except Exception as e: 
        
        pass
 
 
# creating folders to save the images


import os

# Specify the path where you want to create the folders
path = "dataset/images/"

keywords = ["Steven Gerrard"]

# List of folder names you want to create
folders = [name.lower().replace(" ", "_") for name in keywords]

# Iterate over the list of folder names and create them
for folder in folders:
    folder_path = os.path.join(path, folder)
    os.makedirs(folder_path, exist_ok=True)  # exist_ok=True prevents an error if the folder already exists
    print(f"Folder '{folder}' created at {folder_path}")
    
 
 
 
# Importing the previously exported file (with links) and running a loop through the download_images function


query_list = [name.lower().replace(" ", "_") for name in keywords]

for query in query_list:

    
    url_data = pd.read_excel(f'excel_links/{query}_links.xlsx')

    #create a list of the urls

    url_list = list(url_data['Image Link'])

    # create a list for the number of images

    image_count = list(range(0,300))

    # create a list for the queries
    

    # download all images from the links

    for (i,f) in zip(url_list,image_count):

        download_images(i,f,query)
        
    print(f"Successfully saved images of {query}")