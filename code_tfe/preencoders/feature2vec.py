import numpy as np
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

import torch
from torch import nn
import torch.optim as optim

seed_valeur = 42  # Vous pouvez choisir n'importe quelle valeur

np.random.seed(seed_valeur)
torch.manual_seed(seed_valeur)

###########################################################################################

# Object for storing informations on transformations that are applied to the features
class FeatureTransformInfo(object):

    def __init__(self, is_categorical, transform, output_dim, span_idx):
        self.is_categorical = is_categorical
        self.transform = transform
        self.output_dim = output_dim # output dimension of the transform
        self.span_idx = span_idx # in a vector of the size of the "vocabulary" (=d_categories), the indices that correspond to the categories of a given transformed feature


# Class for extracting categories (element of vocabulary) of the data.
# Essentially, we apply one-hot encoding on categorical variables, and also on numerilcal variables after a discretizeation step
class DataCategorizer(object):

    def __init__(self, n_bins=3, strategy='kmeans'):

        self.n_bins = n_bins # number of bins for dicretizing the numerical features
        self.strategy = strategy
        
        # initialize list for storing all information about tranformations applied on each feature
        self.feature_transform_info_list  = []

    # Fits a one hot encoder on a categorical feature
    # Returns all the necessary information to perform the transform
    def _fit_categorical(self, feature_data):

        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
        ohe.fit(feature_data)
        output_dim = len(ohe.categories_[0]) 
        span_idx = (self.categories_count, self.categories_count+output_dim)

        return FeatureTransformInfo(is_categorical=True,
                                    transform=ohe,
                                    output_dim=output_dim,
                                    span_idx=span_idx)

    # Fits a discretizer on numerical features
    # Returns all the necessary information to perform the transform
    def _fit_numerical(self, feature_data):

        kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='onehot-dense', strategy=self.strategy, subsample=200000)
        kbd.fit(feature_data)
        output_dim = self.n_bins
        span_idx = (self.categories_count, self.categories_count+output_dim)

        return FeatureTransformInfo(is_categorical=False,
                                    transform=kbd,
                                    output_dim=output_dim,
                                    span_idx=span_idx)
    
    # Fits one-hot encoder on each numerical feature and a discretizer on each unmerical feature
    # It appends all information about transforms to a list initialized in the __init__ method
    def fit(self, X, categorical_indicator):

        self.categories_count = 0

        for feature_idx, is_categorical in enumerate(categorical_indicator):
            
            feature_data = X[:, feature_idx].reshape(-1,1)

            if is_categorical:
                feature_transform_info = self._fit_categorical(feature_data)
            else:
                feature_transform_info = self._fit_numerical(feature_data)
            
            self.feature_transform_info_list.append(feature_transform_info)
            
            self.categories_count += feature_transform_info.output_dim

    # Given information about a transform, applies the transform to a categorical feature (here, one-hot encoding)
    # Returns the transformed feature; an array of size (n, d) where n is the number of sample and d=sum_j(d_j) is the "vocabulary size" (sum of output dim of each transformations)
    def _transform_categorical(self, feature_data, feature_transform_info):

        n_samples = feature_data.shape[0]
        transformed_data = np.zeros((n_samples, self.categories_count))

        feature_transform = feature_transform_info.transform
        start_idx, end_idx = feature_transform_info.span_idx
        transformed_data[:, start_idx:end_idx] = feature_transform.transform(feature_data)

        return transformed_data
    
    # Given information about a transform, applies the transform to a numerical feature (here, one-hot encoding after discretization)
    # Returns the transformed feature; an array of size (n, d) where n is the number of sample and d=sum_j(d_j) is the "vocabulary size" (sum of output dim of each transformations)
    def _transform_numerical(self, feature_data, feature_transform_info):

        print(feature_data.shape)

        n_samples = feature_data.shape[0]
        transformed_data = np.zeros((n_samples, self.categories_count))
        feature_transform = feature_transform_info.transform
        start_idx, end_idx = feature_transform_info.span_idx
        transformed_data[:, start_idx:end_idx] = feature_transform.transform(feature_data)
        
        return transformed_data

    # Applies transform to every feature according to the information stored in self.feature_transform_info_list
    # Returns an array of size (n, m, d) where n is the number of samples, m the number of features and d the "vocabulary size" (sum of output dim of each transformations)
    def transform(self, X):

        n_samples, m_features = X.shape
        X_transformed = np.zeros((m_features, n_samples, self.categories_count))

        for feature_idx, feature_transform_info in enumerate(self.feature_transform_info_list):

            feature_data = X[:,feature_idx].reshape(-1,1)

            if feature_transform_info.is_categorical:
                X_transformed[feature_idx,:,:] = self._transform_categorical(feature_data, feature_transform_info)
            else:
                X_transformed[feature_idx,:,:] = self._transform_numerical(feature_data, feature_transform_info)

        X_transformed = X_transformed.transpose(1,0,2) #(m, n, d) --> (n, m, d)
        

        return X_transformed
            
###########################################################################################

# Embedding matrices for a Skip-gram Word2Vec inspired model (with negative sampling)
class CategoryEmbedding(nn.Module):

    def __init__(self, d_categories, embedding_dim):
        super().__init__()

        # target embedding matrix of size (d, embedding_dim) where d is the "vocabulary size" and embedding dim is the desired dimension for the embeddings
        self.target_emb = nn.Parameter((torch.rand(d_categories, embedding_dim)-0.5)/100)
        # context embedding matrix of size (d, embedding_dim) where d is the "vocabulary size" and embedding dim is the desired dimension for the embeddings
        self.context_emb = nn.Parameter((torch.rand(d_categories, embedding_dim)-0.5)/100)

    # Forward pass for embedding the target category (target element of vocabulary)
    # The input target_cat is a torch.tensor of size (batch_size, 1, d) where m is the number of features, and d the "vocabulary size"
    # The output target_vec is a torch.tensor of size (batch_size, 1, embedding_dim)
    def forward_t(self, target_cat):
        target_vec = torch.matmul(target_cat,self.target_emb)
        return target_vec
    
    # Forward pass for embedding the context categories (context elements in vocabulary)
    # the input context_cat is a torch.tensor of size (batch_size, m, d) where m is the number of features, and d the "vocabulary size"
    # The output context_vec is a torch.tensor of size (batch_size, m, embedding_dim)
    def forward_c(self, context_cat):
        context_vec = torch.matmul(context_cat,self.context_emb)
        return context_vec
    
    # Forward pass for embedding the negative context categories (elements in vocabulary that are not in the context of the target element)
    # the input negative_cat is a torch.tensor of size (batch_size, m*k, d) where m is the number of features, k is a fixed coefficient, and d the "vocabulary size"
    # The output negative_vec is a torch.tensor of size (batch_size, m*k, embedding_dim)
    def forward_n(self, negative_cat):
        negative_vec = torch.matmul(negative_cat,self.context_emb)
        return negative_vec


# Loss for a Skip-gram Word2Vec inspired model (with negative sampling)    
class NegativeSamplingLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    # forward pass for computing the loss
    # returns the loss averaged over the batches
    def forward(self, target_vec,  context_vec, negative_vec):

        # (batch_size, 1, embedding_dim) --> (batch_size, embedding_dim, 1)
        target_vec = target_vec.transpose(2, 1) 
    
        # context log-sigmoid loss
        # (batch_size, m_features, embedding_dim) x (batch_size, embedding_dim, 1) --> (batch_size, m_features, 1)
        pos_loss = torch.bmm(context_vec, target_vec).sigmoid().log() 
        pos_loss = pos_loss.squeeze().sum(1)  # sum the losses over the sample of c_vectors
        
        # negative log-sigmoid loss
        # (batch_size, m_features*k, embedding_dim) x (batch_size, embedding_dim, 1) --> (batch_size, k*m_features, 1)
        neg_loss = torch.bmm(negative_vec.neg(), target_vec).sigmoid().log()
        neg_loss = neg_loss.squeeze().sum(1)  # sum the losses over the sample of n_vectors

        # return average batch loss
        return -(pos_loss + neg_loss).mean()

###########################################################################################

# Rlass for sampling the triplet (target_cat, context_cat, negative_cat)       
class BatchSampler(object):

    def __init__(self, data, data_transform_info, batch_size, k_negative):
       
        self.data = data
        self.data_tranform_info = data_transform_info
        self.batch_size = batch_size
        self.k_negative = k_negative # coefficient for fixing how many negative examples we use

    # Allows to easily find which category indice corresponds to which feature 
    # Returns a list of length m_features where each element j is itself a list containing the indices of the categories corresponding to feature j
    def _get_categories_per_feature(self,):
         
        _, m_features, _ = self.data.shape

        cat_per_feature = [[i for i in range(*self.data_tranform_info[j].span_idx)] for j in range(m_features)]

        return cat_per_feature
    
    # Allows to easily find the indices of the categories expressed in a a given sample
    # Returns a list of size n_samples where each element j is itself a list containing the indices of the categories expressed in sample j
    def _get_categories_per_sample(self,):

        n_samples, m_features, _ = self.data.shape

        cat_per_sample = [[[] for i in range(m_features)] for j in range(n_samples)]
        for [sample_idx, feature_idx, category_idx] in np.argwhere(self.data>0):
            cat_per_sample[sample_idx][feature_idx].append(category_idx)

        return cat_per_sample
    

    # Allows to find the probability that each category is expressed for each feature 
    # Returns a list of size m_features where each element j is an array containing the log probabilies that each category of feature j is expressed in the dataset (normalized frequencies)
    def _get_category_probabilities_per_feature(self,):

        _, m_features, _ = self.data.shape

        # (n_samples, m_features, d_categories)  --> (n_samples, d_categories)
        data = np.sum(self.data, axis=1)
        # (d_categories, n_samples) x (n_samples, d_categories) --> (d_categories, d_categories)
        counts = data.T @ data 
        # An element ij in the matrix of counts gives the number of times in the dataset that category i and category j appear in the same sample.
        # The element on the diagonal of the count matrix (i=j) give the number of occurence of each category in the dataset

        cat_probs_per_features = [None]*m_features
        for feature_idx in range(m_features):
            start_idx, end_idx = self.data_tranform_info[feature_idx].span_idx
            cat_counts = np.sum(counts[start_idx:end_idx,start_idx:end_idx], axis=0) # this retrives the diagonal elements in the part of the matrix that corresponds to the current feature
            cat_log_counts = np.log(cat_counts+1) # +1 shift to avoid negative probabilities (log(x) with x<1)
            cat_probs = cat_log_counts/np.sum(cat_log_counts) # normalization 
            cat_probs_per_features[feature_idx] = cat_probs

        return cat_probs_per_features
    
    # Allows to easily find the indices of the samples that express a given sample category
    # Returns a list of size d_categories where each element j is itself a list containing the indices of the samples that express category j
    def _get_samples_per_category(self,):

        _, _, d_categories = self.data.shape

        # (n_samples, m_features, d_categories)  --> (n_samples, d_categories)
        data = np.sum(self.data, axis=1)

        samples_per_category = [[] for i in range(d_categories)]
        for [category_idx, sample_idx] in np.argwhere(data.T>0):
            samples_per_category[category_idx].append(sample_idx)

        return samples_per_category
    
    # Samples a batch of the triplet (target_cat, context_cat, negative_cat)
    def sample(self,):

        cat_per_feature = self._get_categories_per_feature()
        cat_probs_per_features = self._get_category_probabilities_per_feature()
        samples_per_category = self._get_samples_per_category()

        _, m_features, d_categories = self.data.shape

        km_negative_cat = self.k_negative*m_features

        # (n_samples, m_features, d_categories)  --> (n_samples, d_categories)
        data = np.sum(self.data, axis=1)
        # (d_categories, n_samples) x (n_samples, d_categories) --> (d_categories, d_categories)
        counts = data.T @ data
    
        target_cat = np.zeros((self.batch_size, 1, d_categories))
        context_cat = np.zeros((self.batch_size, m_features, d_categories))
        negative_cat = np.zeros((self.batch_size, km_negative_cat, d_categories))

        for i in range(self.batch_size):

            # select a feature randomly
            feature_idx = np.random.choice(m_features)
            # select a target category according the the frequency of occurence in the dataset (probability that the target category is epressed)
            target_cat_idx = np.random.choice(cat_per_feature[feature_idx], 
                                              p=cat_probs_per_features[feature_idx])
            # select a sample that expresses this target category
            sample_idx = np.random.choice(samples_per_category[target_cat_idx])
            # select categories that do not occure with the target category
            negative_cat_idx = np.random.choice(np.where(counts[target_cat_idx,:]==0)[0], km_negative_cat)

        
            target_cat[i,:,:] = self.data[sample_idx, feature_idx, :]
            context_cat[i,:,:] = self.data[sample_idx, :, :]
            negative_cat[i, np.arange(km_negative_cat), negative_cat_idx] = 1


        return target_cat, context_cat, negative_cat
    
###########################################################################################

# The preencoder object that puts everything together 
# first we categorize the dataset, then we train the skip-gram word2vec model and extract the embeddings
class Feature2VecPreEncoder(object):

    def __init__(self, embedding_dim=8, batch_size=128, lr=0.1, k_negative=10, n_iter=300):

        self.embedding_dim  = embedding_dim
        self.batch_size = batch_size
        self.lr = lr
        self.k_negative = k_negative
        self.n_iter = n_iter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fits the DataCategorizer, then the skip-gram word2vec model
    # extracts the embeddings
    def fit(self, X, categorical_indicator):

        # categorize data (both categorical and numerical features)
        self.data_categorizer = DataCategorizer()
        self.data_categorizer.fit(X, categorical_indicator)
        X_cat = self.data_categorizer.transform(X)
        # total number of categories ("vocabulary size")
        d_categories = X_cat.shape[2]

        # initialize the batchsampler
        batch_sampler = BatchSampler(data=X_cat,
                                     data_transform_info=self.data_categorizer.feature_transform_info_list,
                                     batch_size=self.batch_size,
                                     k_negative=self.k_negative)
        
        # initialize the skip-gram Word2vec model
        embedding_model = CategoryEmbedding(d_categories=d_categories,
                                            embedding_dim=self.embedding_dim)
        embedding_model = embedding_model.to(self.device)
        criterion = NegativeSamplingLoss()
        criterion = criterion.to(self.device)

        # define the optimizer
        optimizer = optim.Adam(embedding_model.parameters(), lr = self.lr)

        # Training loop
        for i in range(self.n_iter):

            # sample target category, context categories and negative categories
            target_cat, context_cat, negative_cat = batch_sampler.sample()

            # transform input to torch.tensors
            target_cat = torch.from_numpy(target_cat.astype('float32')).to(self.device)
            context_cat = torch.from_numpy(context_cat.astype('float32')).to(self.device)
            negative_cat = torch.from_numpy(negative_cat.astype('float32')).to(self.device)

            # forward pass - embedd target_cat, context cat, and negative cat
            target_vec = embedding_model.forward_t(target_cat)
            context_vec = embedding_model.forward_c(context_cat)
            negative_vec = embedding_model.forward_t(negative_cat)

            # forward pass - compute the loss
            loss = criterion(target_vec, context_vec, negative_vec)

            # backward pass - compute the gradients and update embedding matrices
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%50==0:
                print(f'Iter {i}: Negative Sampling Loss = {loss.item():.4f}')
        
        # extract embeddings
        self.embeddings = embedding_model.target_emb.detach().cpu().numpy()

    # Transforms a dataset with Feature2Vec preencoder.
    # First, the data is transformed using the fitted data_categorizer
    # Then, we replace each category by its embeddind
    # Returns a list of size m_features where each element j of the list is an array of size (n_samples, embedding_dim) which contains the embeddings for feature j (for all samples)
    def transform(self, X):

        X_cat = self.data_categorizer.transform(X)
        _, m_features, _ = X_cat.shape

        X_emb = np.matmul(X_cat, self.embeddings)
        X_emb = X_emb.transpose(1, 0, 2)

        transformed_features =  [X_emb[j,:,:] for j in range(m_features)]

        return transformed_features
