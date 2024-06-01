
from datasets import OpenmlDataset, OpenmlDatasetLoader
from datasets import dataset_train_test_split
from preencoder import PreEncoder 
from models import ToyModel          
from sklearn.metrics import mean_squared_error, accuracy_score
import random
import numpy as np

### example task_ids ###

# 361066 bank-marketing
# 361076 wine_quality
# 361089 california
# 361110 electricity
# 361112 KDDCup09_upselling
# 361116 compass
# 361099 Bike_Sharing_Demand
# 361102 house_sales


ids = [361066,361076,361085,361088,361089,361110,361111,361112,361114,361116,361099,361102]


method = ["baseline","numEncoder_Encoding","feature2vec"]
random_seeds = [42, 123, 456, 789, 101112]

results = {}
std = {}

for task_id in ids :
    results[task_id] = {}
    std[task_id] = {}

    dataset_loader = OpenmlDatasetLoader()
    dataset = dataset_loader.load(task_id)
    dataset.print_info()

   

    for m in method : 

        performances = []

        for seed in random_seeds:

            random.seed(seed)
            np.random.seed(seed)
            
            dataset_train, dataset_test = dataset_train_test_split(dataset, frac=0.7,seed_value=seed)

            preencoder = PreEncoder(method=m)
            preencoder.fit(dataset_train)
            
            dataset_val, dataset_test = dataset_train_test_split(dataset_test, frac=1/3, seed_value=seed)

            aggregation = 'sample'

            force_independence = False
            
            if m == "feature2vec" :
                force_independence = False
            else:
                force_independence = True


            X_train, y_train = preencoder.transform(dataset_train, aggregation=aggregation, force_independence=force_independence)
            X_val, y_val = preencoder.transform(dataset_val, aggregation=aggregation, force_independence=force_independence)
            X_test, y_test = preencoder.transform(dataset_test, aggregation=aggregation, force_independence=force_independence)


            #dataset_train.print_info()
            #dataset_val.print_info()
            #dataset_test.print_info()

            model = ToyModel(task = dataset.task, 
                            n_epochs = 50, 
                            network = aggregation, 
                            m_features = len(dataset.categorical_indicator), 
                            h_dim=256, 
                            early_stopping=False)

            model.fit(X_train, y_train, X_val, y_val)

            #prediction and evaluation
            y_pred = model.predict(X_test)

            #different evaluation metrics 
            if dataset.task == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                performances.append(accuracy*100)
            else:
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                performances.append(rmse)
        std[task_id][m] = round(np.std(performances),3)
        results[task_id][m] = round(np.mean(performances),3)

print(f"results {results}")
print(f"std {std}")
