
#To be implemented with the word2vec like principle

class Feature2VecPreEncoder(object):

    def __init__(self,dataset):
        self.vocab = []
        self.target = []
        self.context = []

    def fit(self, X):
        # Avoir une liste de voisins pour chaque element d'une feature
        # fixer le facteur k, pour la création d'exemple negatifs, le nombre d'epochs 
        # du nn utilisé et la technique d'optimisation
        # fenetre contextuelle du modele ( context 4 = 2 gauche / 2 droite)
        
        pass

    def transform(self, X):
        pass
