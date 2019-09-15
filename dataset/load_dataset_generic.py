from .database_connection import collection
from .news_model import NewsObject
import torch
import gensim.models.keyedvectors as word2vec
import re
from torchtext import data
from torchtext.vocab import Vectors, GloVe, FastText

def extract_words(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text

def load(embedding):
    TEXT = data.Field(sequential=True, tokenize=extract_words, lower=True, include_lengths=True, batch_first=True)
    LABEL = data.LabelField(dtype=torch.float)

    examples = []
    for document in collection.find():
        example = data.Example.fromdict(document, fields={'content': ('content', TEXT), 'label': ('label', LABEL)})
        examples.append(example)
    dataset = data.Dataset(examples, [('content', TEXT), ('label', LABEL)])

    train_data, test_data = dataset.split(stratified=True, split_ratio=0.8)

    if embedding == 'glove':
        TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    elif embedding == 'fasttext':
        TEXT.build_vocab(train_data, vectors=FastText(language='en'))
    elif embedding == 'word2vec':
        # w2vmodel = word2vec.KeyedVectors.load_word2vec_format('.word2vec_cache/GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)
        # w2vmodel.wv.save_word2vec_format('.word2vec_cache/embeddings.vec')
        # w2vmodel = None
        word2vectors = Vectors(name='embeddings.vec', cache='.word2vec_cache')
        TEXT.build_vocab(train_data, vectors=word2vectors)

    LABEL.build_vocab(train_data)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_data, valid_data = train_data.split(stratified=True, split_ratio=0.8) # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data), batch_size=4, sort_key=lambda x: len(x.content), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)

    return TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter