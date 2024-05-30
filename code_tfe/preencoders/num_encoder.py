
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
    def _fit_numerical(self, feature_data):
        
        if feature_data.dtype == np.object_:
            num_tensor = torch.from_numpy(feature_data.astype(float))
        else:
            num_tensor = torch.tensor(feature_data, dtype=torch.float32)

        # randomly fixed number of bins at 8
        bins = compute_bins(num_tensor, n_bins=8)
    
        emb = PiecewiseLinearEncoding(bins)
        output_dim = bins[0].shape[0]
    
        return FeatureTransformInfo(is_categorical=False,
                                    transform=emb,
                                    output_dim=output_dim)

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

            # Special case for dataset with categorical features
            if feature_data.dtype == np.object_ :
                if feature_transform_info.is_categorical != True:
                    feature_data = feature_data.astype(float)
                    tmp = np.array(feature_data,dtype=float)
                    data_tensor = torch.from_numpy(tmp.astype("float32"))
            else:
                data_tensor = torch.tensor(feature_data, dtype=torch.float32)

            #Different cases for the transformation(categorical or numerical)
            if isinstance(feature_transform, OneHotEncoder):
                transformed_feature = feature_transform.transform(feature_data)
                transformed_features.append(transformed_feature)
            else:
                transformed_feature = feature_transform(data_tensor)
                transformed_features.append(transformed_feature.detach().numpy())
            
        return transformed_features