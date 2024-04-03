
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
import torch
import torch.nn as nn
from rtdl_num_embeddings import (
    LinearReLUEmbeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
    compute_bins,
)


class FeatureTransformInfo(object):

    def __init__(self, is_categorical, transform, output_dim):
        self.is_categorical = is_categorical
        self.transform = transform
        self.output_dim = output_dim

class NumPreEncoder(object):

    def __init__(self,):

        self.feature_transform_info_list  = []
    

    # encoding of the ctaegrocial feature,using a one hot encoding 
    def _fit_categorical(self, feature_data):

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(feature_data)
        output_dim = len(ohe.categories_[0]) # number of identified categories

        return FeatureTransformInfo(is_categorical=True,
                                    transform=ohe,
                                    output_dim=output_dim)


    # simple normalisation of the numerical features
    def _fit_numerical(self, feature_data):

        num_tensor = torch.tensor(feature_data, dtype=torch.float32)
        bins = compute_bins(num_tensor)
        emb = PiecewiseLinearEmbeddings(bins,8,activation=False)

        new_rep = emb(num_tensor)

        return FeatureTransformInfo(is_categorical=False,
                                    transform=emb,
                                    output_dim=8)

    def fit(self, X, categorical_indicator):

        for feature_idx, is_categorical in enumerate(categorical_indicator):
            
            feature_data = X[:, feature_idx].reshape(-1,1)

            if is_categorical:
                feature_transform_info = self._fit_categorical(feature_data)
            else:
                feature_transform_info = self._fit_numerical(feature_data)
            
            self.feature_transform_info_list.append(feature_transform_info)
    
    def transform(self, X):

        transformed_features = []

        for feature_idx, feature_transform_info in enumerate(self.feature_transform_info_list):

            feature_data = X[:,feature_idx].reshape(-1,1)
            feature_transform = feature_transform_info.transform
            feature_data_tensor = torch.tensor(feature_data, dtype=torch.float32)
            transformed_feature = feature_transform(feature_data_tensor)
            transformed_features.append(transformed_feature.detach().numpy())
        print(len(transformed_features[0][0][0]))
        #transformed_features = transformed_features.transpose(1,0,2) #(m, n, d) --> (n, m, d)
        return transformed_features
