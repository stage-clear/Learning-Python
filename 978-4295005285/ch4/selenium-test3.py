from selenium import webdriver

# ↓明示的待機のためのインポート
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

url = "http://webscrapingfordatascience.com/complexjavascript/"

driver = webdriver.Chrome('./chromedriver')

driver.get(url)

quote_elements = WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located(
        (By.CSS_SELECTOR, ".quote:not(.decode)")
    )
)

for quote in quote_elements:
    print(quote.text)

input("Press ENTER ot close the automated browser")
driver.quit()
