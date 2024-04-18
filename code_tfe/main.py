
from datasets import OpenmlDataset, OpenmlDatasetLoader
from datasets import dataset_train_test_split
from preencoder import PreEncoder 
from models import ToyModel          
from sklearn.metrics import mean_squared_error, accuracy_score

### example task_ids ###

# 361066 bank-marketing
# 361076 wine_quality
# 361085 sulfur
# 361088 superconduct
# 361089 california
# 361110 electricity
# 361111 eye_movements
# 361112 KDDCup09_upselling
# 361114 rl
# 361116 compass
# 361099 Bike_Sharing_Demand
# 361102 house_sales
# ,361076,361085,361088,361089,361110,361111,361112,361114,361116, 361099,361102


# Num encoding PLE ENCODING memory error for { 361088 (79 features); Aggregation: feature; force_independence: True;} 
# Probleme aussi quand il y'a des features categorielles pour PLE EMBEDDING

""" Faire deja les experiments et recolter les résultats 
- Pour chaque dataset, pour chaque méthode de pre-encoding, pour chaque méthode de pre-encoding, pour chaque taille d'embedding,
- Faire une moyenne des résultats pour chaque dataset
- Faire une moyenne des résultats pour chaque méthode de pre-encoding
- Faire une moyenne des résultats pour chaque taille d'embedding

 """

ids = [361066,361076,361085,361088,361089,361110,361111,361112,361114,361116,361099,361102]

#aggregation_method = ["feature","sample"]

method = ["baseline","numEncoder_encoding","numEncoder_Embeddings","feature2vec"]

emb_size = [8,20,50,100,180,200]

relative_perf ={}
results = {}

for task_id in ids :
    results[task_id] = {}

    dataset_loader = OpenmlDatasetLoader()
    dataset = dataset_loader.load(task_id)
    dataset.print_info()

    dataset_train, dataset_test = dataset_train_test_split(dataset, frac=0.7)

    for m in method : 
        
        preencoder = PreEncoder(method=m)
        preencoder.fit(dataset_train)
        
        dataset_train, dataset_val = dataset_train_test_split(dataset_train, frac=0.7)

        aggregation = 'feature'

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
            results[task_id][m] = (dataset.task,accuracy*100)
        else:
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            results[task_id][m] = (dataset.task,rmse)

print(results)
