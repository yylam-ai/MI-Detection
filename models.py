import os
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

from types import SimpleNamespace


import torch
import numpy as np

from torch_models.dp_classifier import DpClassifierTorchModel
from torch_models.cnn import PyTorchCNNWrapper
from torch_models.nas_cnn import OptiCNNTorchModel
from torch_models.gnn import GNNWrapperTorchModel

os.environ["KMP_WARNINGS"] = "0"

Scoring = {'AUC':'roc_auc', 'Accuracy':'accuracy', 'Recall': 'recall', 'F1-Score': 'f1', 'Precision': 'precision'}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def GNN_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
    print('...Training GNN...')

    edge_list = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), # 2CH neighbours
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), # 4CH neighbours
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)
    ]
    edge_index_list = []
    for u, v in edge_list:
        edge_index_list.append([u, v])
        edge_index_list.append([v, u]) # Add reverse edge for undirected graph
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    model = GNNWrapperTorchModel(
        edge_index=edge_index,
        num_node_features=1,
        num_classes=1,
        verbose=0
    )

    param_grid = {
        'hidden_channels': [16, 32],
        'lr': [0.001, 0.0005],
        'epochs': [20, 50 ,80],
        'batch_size': [16, 32],
        'weight_decay': [5e-4, 1e-4]
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

    grid_search.fit(X_train, y_train)

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