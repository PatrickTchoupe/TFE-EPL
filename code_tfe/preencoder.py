
import numpy as np
from sklearn.preprocessing import  LabelEncoder

from preencoders.baseline import BaselinePreEncoder
from preencoders.feature2vec import Feature2VecPreEncoder
from preencoders.num_encoder import NumPreEncoder

class PreEncoder(object):

    def __init__(self, method='baseline'):
        self.method = method
    

    def fit(self, dataset,emb=8):
        
        # fit pre-encoder for feature pre-encoding
        if self.method == 'baseline':
            self.features_preencoder = BaselinePreEncoder()
            self.features_preencoder.fit(dataset.X.values, dataset.categorical_indicator)
        elif self.method == 'feature2vec':
            self.features_preencoder = Feature2VecPreEncoder(embedding_dim=emb)
            self.features_preencoder.fit(dataset.X.values, dataset.categorical_indicator)
        elif self.method == 'numEncoder':
            self.features_preencoder = NumPreEncoder()
            self.features_preencoder.fit(dataset.X.values, dataset.categorical_indicator)
        else:
            raise ValueError("Unknown pre-encoding method")
        
        # fit pre-encoder for target pre-encoding
        if dataset.task == "regression":
            self.target_preencoder = None 
            # if numerical target is transformed, do not omit to inverse transform before evaluation
        if dataset.task == "classification":
            self.target_preencoder = LabelEncoder()
            self.target_preencoder.fit(dataset.y.values)

    def aggregate_feature_data(self, X, aggregation, force_independence):

        # X is a list of length m where m is the number of features. Each j-th element of the list is an array of size (n, dj)
        # where n is the sample size, and dj is the dimension of the pre-encoding of feature j.

    
        #print(f"size de x0 {X[0].shape}")

        n_samples = X[0].shape[0] # n
        m_features = len(X) # m

        feature_dimensions = []
        for feature_data in X:
            feature_dimensions.append(feature_data.shape[1]) # d_j where j goes from 1 to m


        if aggregation == 'feature':

            if force_independence:

                pre_encoding_dimension = np.sum(feature_dimensions) # in this case, d = sum_j(d_j) 

                X_agg = np.zeros((m_features, n_samples, pre_encoding_dimension)) # shape (m, n, d)

                start_idx = 0
                end_idx = 0
                for j,feature_data in enumerate(X):
                    end_idx += feature_dimensions[j]
                    X_agg[:,:,start_idx:end_idx] = feature_data
                    start_idx = end_idx

                X_agg =  X_agg.transpose(1,0,2) # shape (m, n, d) --> shape (n, m, d)

            else: # in this case, all d_j's must be equal, otherwise np.array(X) will raise a ValueError ! (inhomogeneous dimensions)
                X_agg = np.array(X) # shape (m, n, dj)  , in this case we have dj = d for all j
                X_agg = X_agg.transpose(1, 0, 2) # shape (m, n, d) --> (n, m, d)

        elif aggregation == 'sample':
            X_agg = np.concatenate(X, axis=1) # shape (m, n, dj) --> (n, sum_j(dj)) where j goes from 1 to m

        else:
            raise ValueError("Unknown aggregation")
        return X_agg

    def transform(self, dataset, aggregation='feature', force_independence=True):

        # pre-encoding features
        X = self.features_preencoder.transform(dataset.X.values)
        X = self.aggregate_feature_data(X, aggregation, force_independence)

        # pre-encoding target
        if self.target_preencoder is not None:
            y = self.target_preencoder.transform(dataset.y.values)
        else:
            y = dataset.y.values

        return X, y
    
