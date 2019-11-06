import requests

url = ""

my_cookies = {'PHPSESSID': 'xxxx'}

r = requests.get(url, cookies=my_Cookies)

print(r.text)
