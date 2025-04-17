
import math
import numpy as np

from dpnas.lib.MetaQNN.q_learner import QLearner as QLearner

from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

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

