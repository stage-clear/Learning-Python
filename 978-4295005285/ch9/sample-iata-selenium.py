import pandas
from selenium import webdriver
from selenium.webdriver.support.ui import Select

url = ''

driver = webdriver.Chrome()
driver.implicitly_wait(10)

def get_results (airline_name):
    driver.get(url)

    form_div = driver.find_element_by_css_selector('#aspnetForm .iataStandardForm')
    select = Select(form_div.find_element_by_css_selector('select'))
    select.select_by_value('ByAirlineName')
    text = form_div.find_element_by_css_selector('input[type=text]')
    text.send_keys(airline_name)
    submit = form_div.find_element_by_css_selector('input[type=submit]')
    submit.click()
    table = driver.find_element_by_css_selector('table.datatable')
    table_html = table.get_attribute('outerHTML')
    df = pandas.read_html(str(table_html))
    return df

df = get_results('Lufthansa')
print(df)
