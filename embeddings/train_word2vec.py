from pymongo import MongoClient
import gensim

client = MongoClient()
db = client.fake_news
collection = db.fake_news_corpus


clean_documents = []
for document in collection.find():
    clean_documents.append(gensim.utils.simple_preprocess(document.get('content')))

model = gensim.models.Word2Vec(clean_documents, size=300, window=10, min_count=2, workers=10, iter=10)

model.wv.save_word2vec_format("models/word2vec.model")

print(model.wv.most_similar(positive="dirty"))