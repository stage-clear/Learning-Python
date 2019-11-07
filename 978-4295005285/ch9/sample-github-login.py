import requests
from bs4 import BeautifulSoup

session = requests.Session()

url = 'https://github.com/{}'
username = 'kesuiket'

# ログインページに移動する
r = session.get(url.format('login'))
html_soup = BeautifulSoup(r.text, 'html.parser')

form = html_soup.find(id='login')
print(form)
