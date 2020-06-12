import torch
import gensim.models.keyedvectors as word2vec
import re
from torchtext import data
from torchtext.vocab import Vectors, GloVe

import csv

def extract_words(sentence):
    ignore = ['a', "the", "is"]
    words = re.sub("[^\w]", " ",  sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text

def load(embedding):
    TEXT = data.Field(sequential=True, tokenize=extract_words, lower=True, include_lengths=True, batch_first=True)
    LABEL = data.LabelField(dtype=torch.long)

    examples = []
    with open('ag_news_csv/train.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            document = {
                'content': row[1],
                'label': row[0]
            }
            example = data.Example.fromdict(document, fields={'content': ('content', TEXT), 'label': ('label', LABEL)})
            examples.append(example)
    with open('ag_news_csv/test.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            document = {
                'content': row[1],
                'label': row[0]
            }
            example = data.Example.fromdict(document, fields={'content': ('content', TEXT), 'label': ('label', LABEL)})
            examples.append(example)
    dataset = data.Dataset(examples, [('content', TEXT), ('label', LABEL)])

    if embedding == 'glove_specific':
        vectors = Vectors(name='glove.vec', cache='specific-embeddings')
        TEXT.build_vocab(dataset, vectors=vectors)
    elif embedding == 'glove_generic':
        TEXT.build_vocab(dataset, vectors=GloVe(name='6B', dim=300, cache='.vector_cache'))
    elif embedding == 'fasttext_specific':
        fasttext_vectors = Vectors(name="fasttext.vec", cache="specific-embeddings")
        TEXT.build_vocab(dataset, vectors=fasttext_vectors)
    elif embedding == 'fasttext_generic':
        fasttext_vectors = Vectors(name="crawl-300d-2M.vec", cache=".fasttext_cache")
        TEXT.build_vocab(dataset, vectors=fasttext_vectors)
    elif embedding == 'word2vec_specific':
        word2vectors = Vectors(name='word2vec.vec', cache='specific-embeddings')
        TEXT.build_vocab(dataset, vectors=word2vectors)
    elif embedding == 'word2vec_generic':
        word2vectors = Vectors(name='embeddings.vec', cache='.word2vec_cache')
        TEXT.build_vocab(dataset, vectors=word2vectors)

    LABEL.build_vocab(dataset)

    word_embeddings = TEXT.vocab.vectors
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))

    train_data, test_data = dataset.split(stratified=True, split_ratio=0.8)
    train_data, valid_data = dataset.split(stratified=True, split_ratio=0.8)
    labeled_data, unlabeled_data = train_data.split(stratified=True, split_ratio=0.5) # Further splitting of training_data to create new training_data & validation_data
    labeled_data_iter, unlabeled_data_iter, valid_iter, test_iter = data.BucketIterator.splits((labeled_data, unlabeled_data, valid_data, test_data), batch_size=4, sort_key=lambda x: len(x.content), repeat=False, shuffle=True)

    vocab_size = len(TEXT.vocab)

    return vocab_size, word_embeddings, iter(labeled_data_iter), iter(unlabeled_data_iter), iter(valid_iter), iter(test_iter), len(labeled_data), len(unlabeled_data)