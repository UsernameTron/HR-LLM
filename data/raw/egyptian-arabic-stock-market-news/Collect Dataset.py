import time
from selenium import webdriver
import pandas as pd
from selenium.webdriver.common.by import By

# Define the URL you want to scrape
url = "https://alalamelyoum.co/?s=%D8%AD%D8%AF%D9%8A%D8%AF+%D8%B9%D8%B2"

'''open webdriver => install FoxScroller extension => set speed on 300 => continuous observation  '''

def scroll():
    driver = webdriver.Firefox()
    driver.get(url)
    time.sleep(40)
    for itr in range(20000):
        # Scroll down to the bottom of the page using Selenium
        #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(.5)
        if itr % 25 == 0:
            elements = driver.find_elements(By.XPATH ,"/html/body/div/div/div/div/div/div/div/div/div/ul/li/div/h2/a")
            data = []
            for element in elements:
                data.append({
                    'href': element.get_attribute('href'),
                })
            df = pd.DataFrame(data)
            df.to_csv('D:/temp/Ezz Steel.csv')
        print(itr)
    driver.quit()
    return df

# Scrape content
scroll()


