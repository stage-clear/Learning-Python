# Selenium
- https://selenium-python.readthedocs.io/

```
$ pip3 install selenium
```

- https://sites.google.com/a/chromium.org/chromedriver/downloads
chromedriver はPCにインストールされている Chrome とバージョンを合わせる


## 要素を取得するためのメソッド
- find_element_by_id
- find_element_by_name
- find_element_by_xpath
- find_element_by_link_text
- find_element_by_partial_link_text
- find_element_by_tag_name
- find_element_by_class_name
- find_element_by_css_selector

## 明示的な待機

```python
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
```
