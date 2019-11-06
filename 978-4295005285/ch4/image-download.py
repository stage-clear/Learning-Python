import requests

url = "https://vignette.wikia.nocookie.net/24wikia/images/9/91/Jack_Bauer_%28Day_9%29.jpg/revision/latest/scale-to-width-down/1000?cb=20140502045236"

r = requests.get(url, stream=True)

with open('Jack_Bauer.jpg', 'wb') as f:
    f.write(r.content)
