import openml
import warnings
import numpy as np
import pandas as pd
import random


""" seed_value = 42  # for reproductible results
np.random.seed(seed_value)
random.seed(seed_value) """



# Utility class to display informations about the openml's dataset downloaded 

class OpenmlDataset(object):

    def __init__(self,
                 name, 
                 X, 
                 y, 
                 categorical_indicator, 
                 task):
        
        self.name = name
        self.X = X
        self.y = y
        self.categorical_indicator = categorical_indicator
        self.task = task

    def print_info(self,):
        print('-------------------------------\n'+
        f'Dataset name: {self.name}\n'+ 
        f'n_samples: {self.X.shape[0]}\n'+
        f'm_features: {self.X.shape[1]}'+
        f' (including {np.sum(self.categorical_indicator)} categorical features)\n'+
        f'Task: {self.task}\n'+
        '-------------------------------')


class OpenmlDatasetLoader(object):

    def __init__(self,):
        pass

    def load(self, task_id):

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            task = openml.tasks.get_task(task_id)
            raw_dataset = task.get_dataset()

        # retrieve name
        name = raw_dataset.name

        # retrieve data
        X, y, categorical_indicator, _ = raw_dataset.get_data(dataset_format="dataframe", 
                                                              target=raw_dataset.default_target_attribute)
        for i,column in enumerate(X.columns):
            if X[column].nunique() == 1:
                X.drop(column, axis=1, inplace=True)
                categorical_indicator.pop(i)
        # retrieve task type
        if task.task_type == "Supervised Regression":
            task = "regression" 
        elif task.task_type == "Supervised Classification":
            task = "classification"
        else:
            raise ValueError("Unknown task_type for OpenML task")
        
        # create curstom dataset object
        dataset = OpenmlDataset(name=name,
                                X=X,
                                y=y,
                                categorical_indicator=categorical_indicator,
                                task=task)
        
        return dataset

# Perform a split on the dataset with 70% for the training data
def dataset_train_test_split(dataset, frac=0.7,seed_value=42):

    np.random.seed(seed_value)
    random.seed(seed_value)

    X_train = dataset.X.sample(frac=frac,random_state=seed_value)
    y_train = dataset.y.loc[X_train.index]
    X_test = dataset.X.drop(X_train.index)
    y_test = dataset.y.drop(X_train.index)

    dataset_train = OpenmlDataset(name=f'{dataset.name}_train',
                                  X=X_train,
                                  y=y_train,
                                  categorical_indicator=dataset.categorical_indicator,
                                  task=dataset.task)
    
    dataset_test = OpenmlDataset(name=f'{dataset.name}_test',
                                 X=X_test,
                                 y=y_test,
                                 categorical_indicator=dataset.categorical_indicator,
                                 task=dataset.task)
    
    return dataset_train, dataset_test




