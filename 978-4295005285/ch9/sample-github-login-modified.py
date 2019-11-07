import requests
from bs4 import BeautifulSoup

session = requests.Session()

url = 'https://github.com/{}'
username = 'kesuiket'

r = session.get(url.format('login'))
html_soup = BeautifulSoup(r.text, 'html.parser')

data = {}

for form in html_soup.find_all('form'):
    #非表示のフォームフィールドを取り出す
    for inp in form.select('input[type=hidden]'):
        data[inp.get('name')] = inp.get('value')

# ログイン情報を設定する
data.update({'login': '', 'password': ''})

print('Going to login with the following POST dta: ')
print(data)

if input('Do you want to login (y/n): ') == 'y':
    # ログインを実行する
    r = session.post(url.format('session'), data=data)

    #プロフィールページを取得する
    r = session.get(url.format(username))
    html_soup = BeautifulSoup(r.text, 'html.parser')
    user_info = html_soup.find(class_='vcard-details')
    print(user_info.text)
