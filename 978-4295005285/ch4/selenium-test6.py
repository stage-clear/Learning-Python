from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

class at_least_n_elements_found (object):
    def __init__ (self, locator, n):
        self.locator = locator
        self.n = n

    def __call__ (self, driver):
        elements = driver.find_elements(*self.locator)
        if len(elements) >= self.n:
            return elements
        else:
            return False

url = "http://webscrapingfordatascience.com/complexjavascript/"

driver = webdriver.Chrome('./chromedriver')
driver.get(url)

driver.implicitly_wait(10)

div_element = driver.find_element_by_class_name('infinite-scroll')
quotes_locator = (By.CSS_SELECTOR, ".quote:not(.decode)")

nr_quotes = 0

while True:
    # アクションの使用を開始
    action_chain = ActionChains(driver)
    #名言のブロックに移動
    action_chain.move_to_element(div_element)
    #クリックして捜査対象にする
    action_chain.click()
    # Page Downキーを約10秒間押す
    action_chain.send_keys([Keys.PAGE_DOWN for i in range(10)])
    # 以上のアクションを実行する
    action_chain.perform()

    # 少なくともnr_quotes +１個の名言を取得しようとする
    try:
        all_quotes = WebDriverWait(driver, 3).until(
            at_least_n_elements_found(quotes_locator, nr_quotes + 1)
        )
    except TimeoutException as ex:
        #３秒以内に新しい名言が見つからない場合はこれが全部とみなす
        print("... done!")
        break
    #それ以外の場合は、名言カウンターを更新する
    nr_quotes = len(all_quotes)
    print('... now seeing', nr_quotes, "quotes")

# all_quotesにはすべての名言の要素が含まれている
print(len(all_quotes), 'quotes found\n')

for quote in all_quotes:
    print(quote.text)

input('Press ENTER to close the automated browser')
driver.quit()
