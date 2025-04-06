import os
import math
import copy
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.rocket import MultiRocket
from sklearn.linear_model import RidgeClassifierCV

from dpnas.lib.MetaQNN.q_learner import QLearner as QLearner
from dpnas.lib.Models.network import Net, FullNet

from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from types import SimpleNamespace

os.environ["KMP_WARNINGS"] = "0"

Scoring = {'AUC':'roc_auc', 'Accuracy':'accuracy', 'Recall': 'recall', 'F1-Score': 'f1', 'Precision': 'precision'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class CNN1DModel(nn.Module):
    def __init__(self, inpt_dim, kernel_size=5, filter_size=8):
        super().__init__()
        self.inpt_dim = inpt_dim
        self.kernel_size = kernel_size
        self.filter_size = filter_size

        padding_val = (kernel_size - 1) // 2

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=filter_size,
                      kernel_size=kernel_size, padding=padding_val),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=filter_size, out_channels=int(filter_size / 2),
                      kernel_size=kernel_size, padding=padding_val),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.inpt_dim)
            flattened_output = self.conv_layers(dummy_input)
            self.flattened_dim = flattened_output.numel() # Simpler way

        xx = (self.flattened_dim + 1) // 2

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, xx),
            nn.ReLU(),
            nn.Linear(xx, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        elif x.dim() == 3 and x.shape[1] != 1:
             raise ValueError(f"Input tensor should have 1 channel, but got {x.shape[1]}")

        x = x.to(next(self.parameters()).device)

        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class PyTorchCNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, inpt_dim, kernel_size=5, filter_size=8, learning_rate=1e-1,
                 epochs=50, batch_size=64, verbose=0, device=None):
        self.inpt_dim = inpt_dim
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device

        self.model_ = None
        self.optimizer_ = None
        self.criterion_ = None
        self.classes_ = None
        self.device_ = None 

    def _determine_device(self):
        if self.device:
            return torch.device(self.device)
        else:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        self.device_ = self._determine_device()
        if self.verbose > 0:
            print(f"Using device: {self.device_}")

        # Store classes found in y
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
             raise ValueError("This wrapper currently supports binary classification only.")

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(self.device_)

        self.model_ = CNN1DModel(
            inpt_dim=self.inpt_dim,
            kernel_size=self.kernel_size,
            filter_size=self.filter_size
        ).to(self.device_)

        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        self.criterion_ = nn.BCELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(self.device_), batch_y.to(self.device_)

                self.optimizer_.zero_grad()

                outputs = self.model_(batch_X)

                loss = self.criterion_(outputs, batch_y)

                loss.backward()
                self.optimizer_.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / num_batches
            if self.verbose > 1 and (epoch + 1) % 10 == 0: # Print every 10 epochs if verbose > 1
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}')
            elif self.verbose == 1 and (epoch + 1) == self.epochs: # Print final loss if verbose == 1
                 print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}')

        return self

    def predict_proba(self, X):
        if not self.model_:
            raise RuntimeError("Model has not been fitted yet.")

        self.model_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_)

        all_probas = []
        num_samples = X_tensor.shape[0]
        num_batches = math.ceil(num_samples / self.batch_size)

        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                batch_X = X_tensor[start_idx:end_idx]

                outputs = self.model_(batch_X)
                all_probas.append(outputs.cpu().numpy()) # Move back to CPU for numpy

        probas_pos_class = np.vstack(all_probas) # Shape: (n_samples, 1)

        probas_neg_class = 1.0 - probas_pos_class
        proba_matrix = np.hstack((probas_neg_class, probas_pos_class))

        return proba_matrix

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]



class DpClassifierTorchModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, lr=0.01, epochs=10, batch_size=2):
        self.base_model = base_model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
    

    def fit(self, X, y):
        # Create a fresh clone of the classifier for each fit call
        self.model = copy.deepcopy(self.base_model).to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        y_np = np.asarray(y)
        # Store unique classes found in the target variable y
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return torch.argmax(outputs, dim=1).cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Apply softmax to convert logits to probabilities
            probs = F.softmax(outputs, dim=1)
        return probs.cpu().numpy()
  
class OptiCNNTorchModel(BaseEstimator, ClassifierMixin):
  def __init__(self, model, lr=0.01, epochs=10, batch_size=2):
      self.lr = lr
      self.epochs = epochs
      self.model = model
      self.model.to(DEVICE)
      self.batch_size = batch_size
      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
  def fit(self, X, y):
      y_np = np.asarray(y)
      # Store unique classes found in the target variable y
      self.classes_ = np.unique(y_np)
      self.n_classes_ = len(self.classes_)
      
      X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
      y_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)
      dataset = TensorDataset(X_tensor, y_tensor)
      dataloader = DataLoader(dataset, self.batch_size, shuffle=True)

      self.model.train()
      for _ in range(self.epochs):
          for batch_X, batch_y in dataloader:
              batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
              self.optimizer.zero_grad()
              outputs = self.model(batch_X)
              loss = self.criterion(outputs, batch_y)
              loss.backward()
              self.optimizer.step()
      return self

  def predict(self, X):
      X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
      outputs = self.model(X_tensor)
      return torch.argmax(outputs, axis=1).cpu().numpy()

  def predict_proba(self, X):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = self.model(X_tensor)
    return outputs.cpu().numpy() 

def init_dp_classifier_model(X_train: np.ndarray):
   X_train_init = X_train[:2]
   X_train_init = torch.tensor(X_train_init)

   state_string = "[C(1024,5,1,0,1), C(16,3,1,1,0), SM(2)]"
   args = SimpleNamespace(**{"early_stopping_thresh": 0.15, "patch_size": 12, "task": 2, "net": state_string})
   q_learner = QLearner(args, 2, "")
   q_learner.generate_fixed_net_states_from_string(state_string)
   model = Net(q_learner.state_list, 2, X_train_init, 1e-4, 0)
   
   return model

def init_CNN_opti_model(X_train: np.ndarray):
   X_train_init = X_train[:2]
   X_train_init = torch.tensor(X_train_init)

   state_string = "[C(128,1,1,0,1), P(4,4,0,0), C(1024,1,1,0,1), C(32,1,1,0,1), C(128,1,1,0,1), C(256,1,1,0,1), C(32,1,1,0,1), C(512,1,1,0,1), C(1024,1,1,0,1), C(16,1,1,0,1), C(1024,1,1,0,1), C(16,1,1,0,1), SM(2)]"    
   args = SimpleNamespace(**{"early_stopping_thresh": 0.15, "patch_size": 12, "task": 2, "net": state_string})
   q_learner = QLearner(args, 2, "")
   q_learner.generate_fixed_net_states_from_string(state_string)
   model = FullNet(q_learner.state_list, 2, X_train_init, 1e-4, 0)
   
   return model
   

### MODEL TRAINING ###
def dp_classifier_train(model: DpClassifierTorchModel, X_train, y_train: np.ndarray, REFIT: str):
  print('...Training DP Classfier...')
  param_grid = {
      'lr': [0.001, 0.01, 0.1],
      'epochs': [10, 20, 30, 40, 50],
      'batch_size': [1, 2, 4, 8],
  }
  grid_search = GridSearchCV(estimator = model, n_jobs = -1, param_grid = param_grid, scoring = Scoring, refit = REFIT, cv = 5, verbose=3)
  grid_search = grid_search.fit(X_train, y_train)
  
  return grid_search.best_estimator_, grid_search.best_params_
   
def CNN_opti_train(model: OptiCNNTorchModel, X_train, y_train: np.ndarray, REFIT: str):
  print('...Training NAS Optimal CNN...')
  param_grid = {
      'lr': [0.001, 0.01, 0.1],
      'epochs': [10, 20, 30, 40, 50],
      'batch_size': [1, 2, 4, 8],
  }
  grid_search = GridSearchCV(estimator = model, n_jobs = -1, param_grid = param_grid, scoring = Scoring, refit = REFIT, cv = 5, verbose=3)
  grid_search = grid_search.fit(X_train, y_train)
  
  return grid_search.best_estimator_, grid_search.best_params_

def CNN_train(X_train_tensor: torch.Tensor, y_train_tensor: torch.Tensor, REFIT: str):
    print('...Training CNN...')

    model = PyTorchCNNWrapper(
        inpt_dim=X_train_tensor.shape[2],
        kernel_size=5,
        filter_size=8,
        learning_rate=1e-1,
        epochs=50,
        batch_size=64,
        verbose=0
    )

    param_grid = {
        'kernel_size': [3, 5, 7, 9],
        'filter_size': [4, 10, 16],
        'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],
        'epochs': [25, 50, 75, 100]
    }

    # --- GridSearchCV ---
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=Scoring,
        refit=REFIT,
        cv=5,
        n_jobs=-1, 
        verbose=3
    )

    grid_search.fit(X_train_tensor, y_train_tensor)

    return grid_search.best_estimator_, grid_search.best_params_


def SVM_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
  params_grid = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], 'C': [1, 10, 100, 1000]}]                 
  svm_model = GridSearchCV(SVC(), params_grid, n_jobs = -1, scoring = Scoring, refit = REFIT, cv = 5)
  print('SVM Train')
  svm_model.fit(X_train, y_train)
  print('SVM Train Finished')
  
  return svm_model.best_params_, svm_model.best_estimator_


def DT_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
  params_grid = [{'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'], 
                  'max_features': ['auto', 'sqrt', 'log2']}]                 
  
  dt_model = GridSearchCV(DecisionTreeClassifier(), params_grid, n_jobs = -1, scoring = Scoring, refit = REFIT, cv = 5)
  print('DT Train')
  dt_model.fit(X_train, y_train)
  print('DT Train Finished')
  
  return dt_model.best_params_, dt_model.best_estimator_


def RF_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
  params_grid = [{'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],'criterion': ['gini', 'entropy'], 
                  'max_features': ['auto', 'sqrt', 'log2'], 'class_weight': ['balanced', 'balanced_subsample'],
                  'warm_start': [False, True], 'bootstrap': [False, True]}]                 
  
  RF_model = GridSearchCV(RandomForestClassifier(), params_grid, n_jobs = 1, scoring = Scoring, refit = REFIT, cv = 5)
  print('RF Train')
  RF_model.fit(X_train, y_train)
  print('RF Train Finished')
  
  return RF_model.best_params_, RF_model.best_estimator_


def KNN_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
  params_grid = [{'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],'n_neighbors': [5, 10, 15, 20, 25, 30], 
                  'weights': ['uniform', 'distance'], 'p':[1, 2]}]                 
  
  knn_model = GridSearchCV(KNeighborsClassifier(), params_grid, n_jobs = -1, scoring = Scoring, refit = REFIT, cv = 5)
  print('KNN Train')
  knn_model.fit(X_train, y_train)
  print('KNN Train Finished')
  
  return knn_model.best_params_, knn_model.best_estimator_


def mini_rocket_classifier_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
  params_grid = [{"num_kernels": [5000, 10000, 20000], "n_features_per_kernel": [2, 4, 6]}]
  
  mini_rocket = RocketClassifier(rocket_transform='minirocket')
  grid_search = GridSearchCV(mini_rocket, params_grid, n_jobs=-1,  scoring=Scoring, refit=REFIT, cv=5)
  print('MiniRocket Train')
  grid_search.fit(X_train, y_train)
  print('MiniRocket Train Finished')

  return grid_search.best_params_, grid_search.best_estimator_


def multi_rocket_classifier_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
  params_grid = [{"feature_extractor__num_kernels": [1000, 3000, 6250], "feature_extractor__n_features_per_kernel": [2, 4, 6]}]
  
  multi_rocket = Pipeline([
    ("feature_extractor", MultiRocket()),
    ("classifier", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
  ])
  grid_search = GridSearchCV(multi_rocket, params_grid, n_jobs=-1, scoring=Scoring, refit=REFIT, cv=5)
  print('MultiRocket Train')
  grid_search.fit(X_train, y_train)
  print('MultiRocket Train Finished')

  return grid_search.best_params_, grid_search.best_estimator_


def performance_metrics(CM):
    CM = CM.astype('int64')
    TN = CM[0,0]
    FP = CM[0,1]
    FN = CM[1,0]
    TP = CM[1,1]
    
    Sensitivity = (TP / (TP + FN))
    Specificity = (TN / (TN + FP)) 
    Precision = (TP / (TP + FP))
    F1 = (2*TP) / (2*TP + FP +FN)
    beta = 2
    F2 = (1+beta**2) * ((Precision * Sensitivity) / (beta**2 * Precision + Sensitivity))
    ACC = (TP + TN) / (TP + TN + FN +FP)
            
    metrics = [Sensitivity*100, Specificity*100, Precision*100, F1*100, F2*100, ACC*100]
    
    return metrics

import warnings
warnings.filterwarnings("ignore")