from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import os
import shutil

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

driver = webdriver.Chrome(ChromeDriverManager().install())

# PATH = "/Users/macbookpro/Downloads/chromedriver"
# driver = webdriver.Chrome(PATH)

driver.get('https://www.ilovepdf.com/pdf_to_word')

def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'Downloads')
        
def find_element_with_delay(element_id):
    error = True
    while error == True:
        try:
            driver.find_element_by_id(element_id)
            error = False
        except:
            error = True
            
def pdf2word(file_name):
    driver = webdriver.Chrome(ChromeDriverManager().install())
#     PATH = "/Users/apple/Downloads/chromedriver"
#     driver = webdriver.Chrome(PATH)
    
    print("Opening webpage https://www.ilovepdf.com/pdf_to_word...")
    driver.get('https://www.ilovepdf.com/pdf_to_word')
    driver.implicitly_wait(15)
    
    print("Uploading pdf file to convert...")
    driver.find_element_by_id("pickfiles")
    driver.find_element_by_css_selector('input[type="file"]').clear()
    driver.find_element_by_css_selector('input[type="file"]').send_keys(os.getcwd()+f"/{file_name}")
    print("Done.")
    
    print("Converting to word...")
    element = WebDriverWait(driver, 10).until( 
        EC.presence_of_element_located((By.ID, "processTaskTextBtn")) 
    )
    
    element.click()
    print("Done.")
    
    
#     find_element_with_delay("pickfiles")
#     print("found")
#     driver.find_element_by_id("pickfiles").click()
    print("Saving converted file...")
    error = True
    while error == True:
        try:
            driver.find_element_by_id("pickfiles").click()
            error = False
        except:
            error = True
    
    downloads_folder = get_download_path()
    
    filename_docx = file_name.replace(".pdf", ".docx")
    
    while True:
        if os.path.exists(os.path.join(str(get_download_path()), filename_docx)):
            shutil.move(os.path.join(str(get_download_path()), filename_docx), filename_docx)
            break;
    
    print("Done.")