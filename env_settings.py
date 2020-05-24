import torch
import gensim

CUDA_DEVICE = 0

device = torch.cuda.device(CUDA_DEVICE)

def get_embedding_weights(embedding):
    model = None
    if embedding == 'glove_specific':
        model = gensim.models.KeyedVectors.load_word2vec_format('./specific-embeddings/glove.vec')
    elif embedding == 'glove_generic':
        model = gensim.models.KeyedVectors.load_word2vec_format('./.vector_cache/glove.6B.300d.txt')
    elif embedding == 'fasttext_specific':
        model = gensim.models.KeyedVectors.load_word2vec_format('./specific-embeddings/fasttext.vec')
    elif embedding == 'fasttext_generic':
        model = gensim.models.KeyedVectors.load_word2vec_format('./.fasttext-cache/crawl-300d-2M.vec')
    elif embedding == 'word2vec_specific':
        model = gensim.models.KeyedVectors.load_word2vec_format('./specific-embeddings/word2vec.vec')
    elif embedding == 'word2vec_generic':
        model = gensim.models.KeyedVectors.load_word2vec_format('./.word2vec_cache/embeddings.vec')

    return torch.nn.Parameter(torch.FloatTensor(model.vectors), requires_grad=False)