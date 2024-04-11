
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from rtdl_num_embeddings import (
    LinearReLUEmbeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEncoding,
    PiecewiseLinearEmbeddings,
    compute_bins,
)
import numpy as np

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
       r = np.array(self.X[idx])
       return list(r)

class FeatureTransformInfo(object):

    def __init__(self, is_categorical, transform, output_dim):
        self.is_categorical = is_categorical
        self.transform = transform
        self.output_dim = output_dim

class NumPreEncoder(object):

    def __init__(self,batch_size=100):

        self.feature_transform_info_list  = []
        self.batch_size = batch_size

    # encoding of the ctaegrocial feature,using a one hot encoding 
    def _fit_categorical(self, feature_data):

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(feature_data)
        output_dim = len(ohe.categories_[0]) # number of identified categories

        return FeatureTransformInfo(is_categorical=True,
                                    transform=ohe,
                                    output_dim=output_dim)


    # simple normalisation of the numerical features
    def _fit_numerical(self, feature_data,num="encoding"):

        num_tensor = torch.tensor(feature_data, dtype=torch.float32)
        bins = compute_bins(num_tensor)

        #SI on utilise PLEncoding
        if num == "encoding":
            emb = PiecewiseLinearEncoding(bins)
            output_dim = bins[0].shape[0]
        else:
            #SI on utilise PLEmbeddings
            d_embeddings = 8
            emb = PiecewiseLinearEmbeddings(bins,d_embedding=d_embeddings,activation=False)
            output_dim = d_embeddings

        return FeatureTransformInfo(is_categorical=False,
                                    transform=emb,
                                    output_dim=output_dim)

    def fit(self, X, categorical_indicator,num="encoding"):

        for feature_idx, is_categorical in enumerate(categorical_indicator):
            
            feature_data = X[:, feature_idx].reshape(-1,1)

            if is_categorical:
                feature_transform_info = self._fit_categorical(feature_data)
            else:
                feature_transform_info = self._fit_numerical(feature_data,num=num)
            
            self.feature_transform_info_list.append(feature_transform_info)
    
    def transform(self, X):
        
        transformed_features = []

        for feature_idx, feature_transform_info in enumerate(self.feature_transform_info_list):

            feature_data = X[:,feature_idx].reshape(-1,1)
            feature_transform = feature_transform_info.transform
            feature_data_tensor = torch.tensor(feature_data, dtype=torch.float32)
            transformed_feature = feature_transform(feature_data_tensor)

            #print(transformed_feature.shape)
            transformed_features.append(transformed_feature.squeeze().detach().numpy())
        
        #transformed_features = transformed_features.transpose(1,0,2) #(m, n, d) --> (n, m, d)
        return transformed_features
    
    """ def transform(self, X):
        transformed_features = []
        num_samples = X.shape[0]
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, num_samples)
            batch_X = X[start_idx:end_idx]
            transformed_batch = self._transform_batch(batch_X)
            transformed_features.append(transformed_batch)
        return transformed_features

    def _transform_batch(self, X_batch):
        batch_transformed_features = []
        
        for feature_idx, feature_transform_info in enumerate(self.feature_transform_info_list):

            feature_data = X_batch[:,feature_idx].reshape(-1,1)
            feature_transform = feature_transform_info.transform
            feature_data_tensor = torch.tensor(feature_data, dtype=torch.float32)
            transformed_feature = feature_transform(feature_data_tensor)

            #print(transformed_feature.shape)
            batch_transformed_features.append(transformed_feature.squeeze().detach().numpy())
        return np.array(batch_transformed_features)
 """