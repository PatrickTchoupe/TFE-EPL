
from datasets import OpenmlDataset, OpenmlDatasetLoader
from datasets import dataset_train_test_split
from preencoder import PreEncoder 
from models import ToyModel          
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

task_id =  361099 #361076

dataset_loader = OpenmlDatasetLoader()
dataset = dataset_loader.load(task_id)
dataset.print_info()


dataset_train, dataset_test = dataset_train_test_split(dataset, frac=0.7)

preencoder = PreEncoder()
preencoder.fit(dataset_train)

dataset_train, dataset_val = dataset_train_test_split(dataset_train, frac=0.7)

aggregation = 'feature'
force_independence = True


X_train, y_train = preencoder.transform(dataset_train, aggregation=aggregation, force_independence=force_independence)
X_val, y_val = preencoder.transform(dataset_val, aggregation=aggregation, force_independence=force_independence)
X_test, y_test = preencoder.transform(dataset_test, aggregation=aggregation, force_independence=force_independence)

print(X_train.shape)

dataset_train.print_info()
dataset_val.print_info()
dataset_test.print_info()

model = ToyModel(task = dataset.task, 
                 n_epochs = 50, 
                 network = aggregation, 
                 m_features = len(dataset.categorical_indicator), 
                 h_dim=256, 
                 early_stopping=False)

model.fit(X_train, y_train, X_val, y_val)
