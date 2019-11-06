from selenium import webdriver

url = "http://webscrapingfordatascience.com/complexjavascript/"

driver = webdriver.Chrome('./chromedriver')

# 暗黙的待機を設定する
driver.implicitly_wait(10)

driver.get(url)

for quote in driver.find_elements_by_class_name('quote'):
    print(quote.text)

input('Press ENTER to close the automated browser')
driver.quit()
