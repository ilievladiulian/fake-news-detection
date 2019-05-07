from pymongo import MongoClient

client = MongoClient()
db = client.fake_news
collection = db.fake_news_corpus