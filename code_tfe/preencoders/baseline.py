from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

# All the code come from the following source:
# edouard.couplet@uclouvain.be

class FeatureTransformInfo(object):

    def __init__(self, is_categorical, transform, output_dim):
        self.is_categorical = is_categorical
        self.transform = transform
        self.output_dim = output_dim

class BaselinePreEncoder(object):

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

        qt = QuantileTransformer(output_distribution="normal")
        qt.fit(feature_data)
        output_dim = 1

        return FeatureTransformInfo(is_categorical=False,
                                    transform=qt,
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


            transformed_feature = feature_transform.transform(feature_data)
            transformed_features.append(transformed_feature)
            
        return transformed_features
