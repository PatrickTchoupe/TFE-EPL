import numpy as np
import torch

from torch import nn
from torch.utils.data import Dataset, DataLoader

seed_valeur = 42  # Vous pouvez choisir n'importe quelle valeur
np.random.seed(seed_valeur)
torch.manual_seed(seed_valeur)

class MyDataset(Dataset):
    def __init__(self, X, y=None, train=False):

        self.train = train

        self.X = torch.tensor(X, dtype=torch.float32)

        if self.train:
            self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):

        if self.train:
            return self.X[index], self.y[index]
        else:
            return self.X[index]

# MLP like neural network for experiments
           
class ToySampleNN(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(ToySampleNN, self).__init__()
        
        self.enc1 =  nn.Linear(in_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dec = nn.Linear(hidden_dim,1)

    def forward(self, x, task):

        x_enc = self.enc1(x).relu()
        x_enc = self.enc2(x_enc).relu()
 
        x_dec = self.dec(x_enc)
        if task == "classification":
             x_dec = x_dec.sigmoid()

        return x_dec


# Transformer like neural network
    
class ToyFeatureNN(nn.Module):

    def __init__(self, in_dim, m_features, hidden_dim):
        super(ToyFeatureNN, self).__init__()
        
        self.mix1 = nn.Linear(m_features, m_features, bias=False)
        self.enc1 = nn.Linear(in_dim, hidden_dim)  

        self.mix2 = nn.Linear(m_features, m_features, bias=False)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim) 

        self.dec = nn.Linear(hidden_dim,1)

    def forward(self, x, task):

        x_enc = torch.transpose(x,2,1)
        x_enc = self.mix1(x_enc)
        x_enc = torch.transpose(x_enc,2,1)
        x_enc = self.enc1(x_enc).relu()

        x_enc = torch.transpose(x_enc,2,1)
        x_enc = self.mix2(x_enc)
        x_enc = torch.transpose(x_enc,2,1)
        x_enc = self.enc2(x_enc).relu()
        
        x_aggr = torch.sum(x_enc, dim =1)

        x_dec = self.dec(x_aggr)
        if task == "classification":
             x_dec = x_dec.sigmoid()

        return x_dec
    
    
class ToyModel(object):

    def __init__(self, task, n_epochs, network, m_features, h_dim=256, early_stopping=False):
        
        self.task = task
        if self.task == "classification":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.MSELoss()
        self.n_epochs = n_epochs
        self.network = network
        self.m_features = m_features
        self.h_dim = h_dim

        self.early_stopping = early_stopping
        self.device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    def fit(self, x_train, y_train, x_val, y_val):

        train_data = MyDataset(x_train, y_train, train=True)
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)

        val_data = MyDataset(x_val, y_val, train=True)
        val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

        in_dim = x_train.shape[-1]

        if self.network == "sample":
            self.model = ToySampleNN(in_dim=in_dim,
                                     hidden_dim=self.h_dim).to(self.device)
        elif self.network == "feature":
            self.model = ToyFeatureNN(in_dim=in_dim,
                                      m_features=self.m_features, 
                                      hidden_dim=self.h_dim).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()

        if self.early_stopping:
            best_val_loss = float('inf')
            self.best_model_state = None
            patience = 20  # number of epochs to wait for improvement
            wait = 0

        self.train_loss = np.zeros(self.n_epochs)
        self.val_loss = np.zeros(self.n_epochs)

        for epoch in range(self.n_epochs):
            
            #train the model
            running_loss = 0
            for x_train, y_train in train_loader:

                x_train = x_train.to(self.device)
                y_train =  y_train.view(-1,1).to(self.device)
                y_pred = self.model(x_train, self.task)
                loss = self.criterion(y_pred,y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += (loss.item()*len(y_train))
                train_loss = running_loss/len(train_data)
            self.train_loss[epoch] = train_loss

            # Validate the model
            running_loss = 0
            with torch.no_grad():
                for x_val,y_val in val_loader:
                    x_val = x_val.to(self.device)
                    y_val =  y_val.view(-1,1).to(self.device)
                    y_pred = self.model(x_val,self.task)
                    loss = self.criterion(y_pred,y_val).detach().cpu().numpy()

                    running_loss += (loss.item()*len(y_val))
                    val_loss = running_loss/len(val_data)
                self.val_loss[epoch] = val_loss
            
            if self.early_stopping:
                # Check for early stopping
                if  val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print(f"Stopped training after {epoch} epochs.")
                        break
            
            if (epoch+1)%10 == 0:
                print(f'Epoch :{epoch+1} | Train loss: {train_loss:5.3f}| Val loss: {val_loss:5.3f}')
    
    def predict(self, X):
        if self.early_stopping:
            self.model.load_state_dict(self.best_model_state)
            
        self.model.eval()

        eval_data = MyDataset(X, train=False)
        eval_loader = DataLoader(eval_data, batch_size=256, shuffle=False)

        y = []

        with torch.no_grad():
            for x_eval in eval_loader:
                x_eval = x_eval.to(self.device)

                y.append(self.model(x_eval,self.task).detach().cpu())

        y = torch.cat(y, dim=0).numpy()

        if self.task == "classification":
            y = np.where(y>0.5,1,0)

        return y
    
    








