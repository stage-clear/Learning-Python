import requests

url = ""

# まずPOSTリクエストを実行する
r = requests.post(url, data={'username': 'dummy', 'password': '1234'})

# r.headers または r.cookies print(r.cookies) で
# Cookie の値を取得する
my_cookies = r.cookies

# r.cookies は  RequestsCookieJar オブジェクト
# 辞書のようにアクセスできるので、以下でも動作する
my_cookies['PHPSSID'] = r.cookies.get('PHPSESSID')

# 秘密のページへのGETリクエストを実行する
r = requests.get(url + 'secret.php', cookies=my_cookies)

print(r.text)
