import requests
from bs4 import BeautifulSoup
# Webページを取得して解析する

load_url = "https://ui.adsabs.harvard.edu/search/q=transition%20edge%20sensor&sort=date%20desc%2C%20bibcode%20desc&p_=0"
html = requests.get(load_url)
soup = BeautifulSoup(html.content, "html.parser")

text=soup.get_text()
print(text)
