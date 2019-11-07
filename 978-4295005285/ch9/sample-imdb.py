import requests
from bs4 import BeautifulSoup

url = 'http://www.imdb.com/title/tt0285331/episodes'

episodes = []
ratings = []

# シーズン１から７までループする
for season in range(1,8):
    r = requests.get(url, params={'season': season})
    soup = BeautifulSoup(r.text, 'html.parser')
    listing = soup.find('div', class_='eplist')

    for eqnr, div in enumerate(listing.find_all('div', recursive=False)):
        episode = '{}.{}'.format(season, eqnr + 1)
        rating_el = div.find(class_='ipl-rating-star__rating')
        rating = float(rating_el.get_text(strip=True))
        print('Episode: ', episode, '-- rating: ', rating)
        episodes.append(episode)
        ratings.append(rating)
