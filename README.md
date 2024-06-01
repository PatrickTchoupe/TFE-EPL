# TFE-EPL
Repo pour mes travaux de mémoire de master UCLouvain

Titre : Feature embedding techniques for enhancing deep learning models on tabular data

Superviseur : John LEE, Edouard COUPLET

Abstract

Many datasets are stored in tabular form, where rows correspond to samples
and columns to attributes or features. While deep learning models excel in
other data modalities such as images or text, they often underperform when
applied to tabular data compared to traditional models like gradient-boosted
tree ensembles (e.g., XGBoost). This is partly due to the heterogeneous na-
ture of tabular data features. Each feature may represent a different variable,
either categorical or numerical, complicating the learning process for neural
network-based models.
An important step in any deep learning pipeline for tabular data is to
encode categorical features in a way that is usable by the network and to
project all features into a more homogeneous space that is more conducive
to learning. A common approach is to use one-hot encoding for categorical
features and allow the first few layers of a neural network to learn suitable
feature representations in a supervised manner during the training phase.
Self-supervised learning has led to considerable performance gains in the
natural language processing world: a decade ago with static embedding tech-
niques like Word2Vec, and more recently with contextual embedding tech-
niques that are key elements in the success of large language models like GPT.
The goal of this thesis is to explore such self-supervised feature embedding
techniques for tabular data and evaluate their impact on the performance
of simple deep learning architectures (e.g., MLP) and more advanced ar-
chitectures like Transformer-like models for various downstream tasks. The
primary focus is on understanding how these feature embedding techniques
impact the learning process and how they can be leveraged to improve perfor-
mance. We have developed a self-supervised embedding technique inspired
by Word2Vec, and through our research, we have obtained insights that can
aid in the development of better deep learning models for tabular data.
