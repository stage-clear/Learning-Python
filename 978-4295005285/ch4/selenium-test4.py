from selenium import webdriver

# ↓明示的待機のためのインポート
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

url = "http://webscrapingfordatascience.com/complexjavascript/"

driver = webdriver.Chrome('./chromedriver')

driver.get(url)


class at_least_n_elements_found (object):
    def __init__ (self, locator, n):
        self.locator = locator
        self.n = n
    def __call__ (self, driver):
        # ここで何らかの処理を実行し
        # 条件の結果次第で False またはそれ以外の結果を返す
        elements = driver.find_elements(*self.locator)
        if len(elements) >= self.n:
            return elements
        else:
            return False

wait = WebDriverWait(driver, 10)

quote_elements = wait.until(
    at_least_n_elements_found((By.CSS_SELECTOR, '.quote:not(.decode)'), 3)
)

for quote in quote_elements:
    print(quote.text)

input("Press ENTER ot close the automated browser")
driver.quit()
