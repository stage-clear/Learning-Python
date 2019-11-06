import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

base_url = ""
link_seen = set()

def vist (url, links_seen):
    html = requests.get(url).text
    html_soup = BeautifulSoup(html, 'html.parser')
    links_seen.add(url)

    for link in html_soup.find_all('a'):
        link_url = link.get('href')

        if link_url is None:
            continue
        full_url = urljoin(url, link_url)

        if full_url in links_seen:
            continue
        print("Found a new page: ", full_url)
        # 通常は結果をここでも保存する
        visit(full_url, links_seen)
        #!ウェブクローリングに再帰呼び出しを使うのは確実な手法とは言えません

visit(base_url, links_seen)
