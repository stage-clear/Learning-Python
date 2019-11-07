import requests
import dataset
from bs4 import BeautifulSoup
import urllib.parse

db = dataset.connect('sqlite:///quotes.db')

authors_seen = set()

base_url = 'http://quotes.toscrape.com/'

def clean_url(url):
    # '/author/Steve-Martin' を 'Steve-Martin' に変更してクリーンにする
    # urljoin を使って絶対URLにする
    url = urllib.parse.urljoin(base_url, url)
    # urlparse を使ってパスの部分を取り出す
    path = urllib.parse.urlparse(url).path
    # '/' でパスを分割して２番目の部分を取得する
    return path.split('/')[2]

def scrape_quotes (html_soup):
    for quote in html_soup.select('div.quote'):
        quote_text = quote.find(class_='text').get_text(strip=True)
        quote_author_url = clean_url(quote.find(class_='author') \
            .find_next_sibling('a').get('href'))
        quote_tag_urls = [clean_url(a.get('href'))
            for a in quote.find_all('a', class_='tag')]
        authors_seen.add(quote_author_url)

        quote_id = db['quotes'].insert({
            'text': quote_text,
            'author': quote_author_url
        })

        db['quote_tags'].insert_many(
            [{'quote_id': quote_id, 'tag_id': tag} for tag in quote_tag_urls]
        )

def scrape_author(html_soup, author_id):
    author_name = html_soup.find(class_='author-title').get_text(strip=True)
    author_born_date = html_soup.find(class_='author-born-date').get_text(strip=True)
    author_born_loc = html_soup.find(class_='author-born-location').get_text(strip=True)
    author_description = html_soup.find(class_='author-description').get_text(strip=True)

    db['authors'].insert({
        'author_id': author_id,
        'name': author_name,
        'born_date': author_born_date,
        'description': author_description
    })

# まずすべての名言のページをスクレイピングする
url = base_url
while True:
    print('Now scraping page:', url)
    r = requests.get(url)
    html_soup = BeautifulSoup(r.text, 'html.parser')
    scrape_quotes(html_soup)

    next_a = html_soup.select('li.next > a')
    if not next_a or not next_a[0].get('href'):
        break
    url = urllib.parse.urljoin(url, next_a[0].get('href'))

for author_id in authors_seen:
    url = urllib.parse.urljoin(base_url, '/author/' + author_id)
    print('Now scraping author: ', url)
    r = requests.get(url)
    html_soup = BeautifulSoup(r.text, 'html.parser')
    scrape_author(html_soup, author_id)
