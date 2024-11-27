# importing libraries
from pandas.core.frame import DataFrame
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time

# add all racers that you want to scrape images of

keywords = ["Mo Salah","Christiano Ronaldo","Messi","Wayne Rooney"]

# creating the function to scrape link of images to be downloaded

def search_google(search_query):
    
    # breaking the name down
    
    converted_name = search_query.lower().replace(" ", "_")
    
    data_dic = {
                "Keyword":[],
                "Image Link":[]
               }
    
    search_url = f"https://www.google.com/search?q=40+face+potrait+pictures+of+{search_query}&sca_esv=332c1457e26e21ac&rlz=1C5CHFA_enKE1073KE1073&udm=2&biw=1440&bih=812&ei=D2VAZ-vYKaGZi-gPifrQoAE&ved=0ahUKEwirnMqw5e-JAxWhzAIHHQk9FBQQ4dUDCBA&uact=5&oq=200+face+potrait+pictures+of+steven+gerrard&gs_lp=EgNpbWciKzIwMCBmYWNlIHBvdHJhaXQgcGljdHVyZXMgb2Ygc3RldmVuIGdlcnJhcmRI7yBQqQhYrhtwAXgAkAEAmAGOBKAByxCqAQcyLTcuNS0xuAEDyAEA-AEBmAIAoAIAmAMAiAYBkgcAoAfoAg&sclient=img"
    browser = webdriver.Chrome()

    # Open browser to begin search
    browser.get(search_url)
    
    # Maximize the window
    browser.maximize_window()
    
    time.sleep(3)
    
    count = 0
    
    for i in range(0,40):
        
        count += 1
        
        #click the picture
    
        img_box = browser.find_elements(By.CSS_SELECTOR,'div.H8Rx8c')[i]
        
        try:
            img_box.click()
        

            time.sleep(5)
        
            # extract the link
        
        
            img_box_2 = browser.find_elements(By.CSS_SELECTOR,'div.p7sI2')[1]

            link = img_box_2.find_elements(By.TAG_NAME,'img')[0].get_attribute('src')
            
        except:
            
            # if the click gets interecepted
            
            try:
            
                browser.execute_script("window.scrollBy(0, 100);")

                time.sleep(2)

                img_box.click()


                time.sleep(5)

                # extract the link


                img_box_2 = browser.find_elements(By.CSS_SELECTOR,'div.p7sI2')[1]

                link = img_box_2.find_elements(By.TAG_NAME,'img')[0].get_attribute('src')
                
            except:
                
                pass

        # append to a dictionary
        
        
        data_dic["Keyword"].append(keyword)
        data_dic["Image Link"].append(link)
     
    # export the dictionary to an excel
        
        if count % 20 == 0:
    
            df = pd.DataFrame(data=data_dic)

            df.to_excel(f'../excel_links/{converted_name}_links.xlsx', index=False)
            
            print(f"Downloaded {count} images of {search_query}")
    

    
    browser.close()
    
    
# running the seach function that we created above through the keywords

for keyword in keywords:
    print(f"Downloading Images of {keyword}")
    search_google(keyword)