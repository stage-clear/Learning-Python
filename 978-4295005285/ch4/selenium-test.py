from selenium import webdriver
# chromedriver はPCにインストールされている Chrome とバージョンを合わせる

url = "https://24.fandom.com/wiki/Jack_Bauer"

driver = webdriver.Chrome(executable_path='./chromedriver')
driver.get(url)

input('Press Enter to close the automated browser')

driver.quit()
