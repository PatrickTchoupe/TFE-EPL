
#To be implemented with the word2vec like principle
from utils import *


class Feature2VecPreEncoder(object):

    def __init__(self,dataset):
        self.vocab = []
        self.target = []
        self.context = []
        self.model = None

    def fit(self, X, embeddings_size, windows_size):
        # Avoir une liste de voisins pour chaque element d'une feature
        # fixer le facteur k, pour la création d'exemple negatifs, le nombre d'epochs 
        # du nn utilisé et la technique d'optimisation
        # fenetre contextuelle du modele ( context 4 = 2 gauche / 2 droite)

        self.vocab = get_vocab(X)
        self.features_to_id, self.id_to_word = mapping(self.vocab)

        self.target, self.context = generate_training_data(self.vocab,
                                                        self.features_to_id,
                                                        windows_size)
        
        self.model = init_network(len(self.features_to_id), embeddings_size)
        

    def transform(self, X, n_iter, learning_rate):
        history = [backward(self.model, self.target, self.target, learning_rate) for _ in range(n_iter)]
        embeddings = get_embedding(self.model, self.vocab, self.features_to_id)

        return embeddings
