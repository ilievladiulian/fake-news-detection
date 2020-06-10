import torch
import gensim
import torchtext.vocab as vocab

CUDA_DEVICE = 3

device = torch.cuda.device(CUDA_DEVICE)

def get_embedding_weights(embedding):
    embeddings_file = ''
    cache = ''
    if embedding == 'glove_specific':
        embeddings_file='glove.vec'
        cache='specific-embeddings'
    elif embedding == 'glove_generic':
        embeddings_file='glove.6B.300d.txt'
        cache='.vector_cache'
    elif embedding == 'fasttext_specific':
        embeddings_file = 'fasttext.vec'
        cache = 'specific-embeddings'
    elif embedding == 'fasttext_generic':
        embeddings_file = 'crawl-300d-2M.vec'
        cache = '.fasttext_cache'
    elif embedding == 'word2vec_specific':
        embeddings_file = 'word2vec.vec'
        cache = 'specific-embeddings'
    elif embedding == 'word2vec_generic':
        embeddings_file = 'embeddings.vec'
        cache = '.word2vec_cache'

    model = vocab.Vectors(name=embeddings_file, cache=cache)

    return torch.nn.Parameter(torch.FloatTensor(model.vectors), requires_grad=False)