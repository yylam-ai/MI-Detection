import os
import numpy as np
from keras import layers, Model
from keras import optimizers
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.rocket import MultiRocket
from sklearn.linear_model import RidgeClassifierCV

os.environ["KMP_WARNINGS"] = "0"

Scoring = {'AUC':'roc_auc', 'Accuracy':'accuracy', 'Recall': 'recall', 'F1-Score': 'f1', 'Precision': 'precision'}

def cnn1D_model(inpt_dim, kernel_size = 5, filter_size = 8, learning_rate = 1e-1):
    inputs = layers.Input(shape=(inpt_dim, 1))
    x = layers.Convolution1D(filters=filter_size, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    x = layers.Convolution1D(filters=int(filter_size / 2), kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    
    x = layers.Flatten()(x)
    
    flattened_dim = x.shape[1] 
    xx = int(flattened_dim - ((flattened_dim - 1) / 2))

    x = layers.Dense(xx, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    opti = optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = opti, metrics = ['accuracy'])
    
    return model


def CNN_train(X_train: np.ndarray, y_train: np.ndarray, REFIT: str):
    print('...Training...')    
    model = KerasClassifier(build_fn = cnn1D_model, inpt_dim = X_train.shape[1], kernel_size = 5, filter_size = 8, learning_rate = 1e-1) 
    kernel_size = [3, 5, 7, 9, 11, 13, 15]
    filter_size = [4, 8, 12, 16, 24, 32]
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    param_grid = dict(kernel_size = kernel_size, filter_size = filter_size, learning_rate = learning_rate,  epochs = [25, 50, 75, 100])
    grid_search = GridSearchCV(estimator = model, n_jobs = -1, param_grid = param_grid, scoring = Scoring, refit = REFIT, cv = 5)
    grid_search = grid_search.fit(X_train, y_train)
    
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