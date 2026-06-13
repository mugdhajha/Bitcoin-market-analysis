from dotenv import load_dotenv
import os
import requests
import pandas as pd

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

url = (
    "https://newsapi.org/v2/everything?"
    "q=bitcoin OR cryptocurrency"
    "&language=en"
    "&sortBy=publishedAt"
    "&pageSize=100"
    f"&apiKey={API_KEY}"
)

response = requests.get(url)

data = response.json()

articles = []



for article in data["articles"]:
    articles.append({
        "title": article.get("title"),
        "description": article.get("description"),
        "content": article.get("content"),
        "source": article.get("source", {}).get("name"),
        "publishedAt": article.get("publishedAt"),
        "url": article.get("url")
    })

df = pd.DataFrame(articles)

bitcoin_keywords = [
    "bitcoin",
    "btc",
    "etf",
    "blockchain",
    "satoshi",
    "halving",
    "mining"
]

mask = (
    (
        df["title"].fillna("")
        + " "
        + df["description"].fillna("")
    )
    .str.lower()
    .apply(
        lambda x: any(
            word in x
            for word in bitcoin_keywords
        )
    )
)

df = df[mask]

df.to_csv("data/latest_news.csv", index=False)

print(f"Saved {len(df)} articles")