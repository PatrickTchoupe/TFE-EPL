from datasets import OpenmlDataset, OpenmlDatasetLoader
from datasets import dataset_train_test_split
from preencoder import PreEncoder 
import numpy as np
from models import ToyModel    


np.random.seed(42)

def get_vocab(X): 
    features = {}
    for i in X.columns :
        features[i] = set(X[i].values)
    vocab = []
    for k,v in features.items() : 
        for i in v:
            vocab.append((i,k))
    return vocab

def get_embedding(model, X, features_to_id):
        embeddings = {}
        for i in X:
            try:
                idx = features_to_id[i]
            except KeyError:
                print("`feature` not in corpus")
            one_hot = one_hot_encode(idx, len(features_to_id))
            embeddings[i] = forward(model, one_hot)["a1"]
        return embeddings

def init_network(embeddings_size, vocab_size):
    model = {
        "w1": np.random.randn(vocab_size, embeddings_size),
        "w2": np.random.randn(embeddings_size, vocab_size)
    }
    return model

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

def forward(model, X, return_cache=True):
    cache = {}
    
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    
    if not return_cache:
        return cache["z"]
    return cache


def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)

def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

def mapping(tokens):
    word_to_id = {}
    id_to_word = {}
    
    for i, token in enumerate(tokens):
        word_to_id[token] = i
        id_to_word[i] = token
    
    return word_to_id, id_to_word

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res

def generate_training_data(tokens, word_to_id, window):
    X = []
    y = []
    n_tokens = len(tokens)
    
    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i), 
            range(i, min(n_tokens, i + window + 1))
        )
        
        for j in idx:
            if i == j:
                continue
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))
    
    return np.asarray(X), np.asarray(y)