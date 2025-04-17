
import numpy as np

from dpnas.lib.MetaQNN.q_learner import QLearner as QLearner

from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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