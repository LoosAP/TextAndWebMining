import feedparser
import pandas as pd
from bs4 import BeautifulSoup
import re

# RSS feeds
FEEDS = {
    'Telex': 'https://telex.hu/rss',
    'Blikk': 'https://www.blikk.hu/rss/',
    'Portfolio': 'https://www.portfolio.hu/rss/all.xml',
    '24.hu': 'https://24.hu/feed/',
    'hvg.hu': 'https://hvg.hu/rss',
    '444.hu': 'https://444.hu/feed'
}

def clean_html(raw_html):
    """Remove html tags from a string"""
    if not raw_html:
        return ""
    # Use BeautifulSoup to parse and get text
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove leftover non-alphanumeric characters and multiple spaces
    text = re.sub(r'[^a-zA-Z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_and_clean_data():
    """Fetch data from RSS feeds and clean it."""
    articles = []
    for source, url in FEEDS.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.title
            description = clean_html(entry.get('summary', ''))
            articles.append({
                'source': source,
                'title': title,
                'description': description,
                'link': entry.link
            })
    
    df = pd.DataFrame(articles)
    return df

if __name__ == "__main__":
    data = fetch_and_clean_data()
    # Save to a CSV file
    data.to_csv('articles.csv', index=False, encoding='utf-8-sig')
    print("Data fetched and saved to articles.csv")
    print(data.head())
