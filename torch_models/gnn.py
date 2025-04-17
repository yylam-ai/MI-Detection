import torch
import numpy as np

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.base import BaseEstimator, ClassifierMixin

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=1):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels) # Add BatchNorm layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels) # Add BatchNorm layer

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = torch.nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x) # Apply BatchNorm
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x) # Apply BatchNorm
        x = F.relu(x)
        # NO dropout after last GCN layer usually

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training) # Dropout in MLP part is fine
        x = self.lin2(x)
        return x


class HeartDataset(Dataset):
    def __init__(self, features_all, labels_all, edge_index, transform=None, pre_transform=None):
        super().__init__(transform, pre_transform)
        self.features_all = np.asarray(features_all)
        self.labels_all = np.asarray(labels_all)
        self.edge_index = edge_index

    def len(self):
        return len(self.labels_all)

    def get(self, idx):
        patient_features = self.features_all[idx]
        node_features = torch.tensor(patient_features, dtype=torch.float).unsqueeze(1) # Shape [12, 1]
        label = torch.tensor(self.labels_all[idx], dtype=torch.float) # Target label as float for BCEWithLogitsLoss

        graph_edge_index = self.edge_index if self.edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long)
        data = Data(x=node_features, edge_index=graph_edge_index, y=label)
        return data


class GNNWrapperTorchModel(BaseEstimator, ClassifierMixin):
    def __init__(self, edge_index,
                 hidden_channels=32, num_node_features=1, num_classes=1,
                 lr=0.001, epochs=50, batch_size=16, weight_decay=5e-4,
                 random_state=None, verbose=0):
        """
        Wrapper for PyTorch Geometric GNN model compatible with scikit-learn.
        Instantiates the GNN model inside fit().
        """
        self.edge_index = edge_index
        # GNN Params
        self.hidden_channels = hidden_channels
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        # Training Params
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        # Other
        self.random_state = random_state
        self.verbose = verbose # Control printing during fit

        # Internal attributes
        self.model_ = None # Use trailing underscore for fitted attributes (sklearn convention)
        self.criterion_ = None
        self.optimizer_ = None
        self.classes_ = None
        self.n_classes_ = None

    def get_params(self, deep=True):
        return {
            "edge_index": self.edge_index, # Edge index might not be clonable by deepcopy, handle carefully if needed
            "hidden_channels": self.hidden_channels,
            "num_node_features": self.num_node_features,
            "num_classes": self.num_classes,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "verbose": self.verbose
        }

    # This method is required by sklearn for setting parameters during grid search
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):

        self.model_ = GNN(
            num_node_features=self.num_node_features,
            hidden_channels=self.hidden_channels,
            num_classes=self.num_classes
        ).to(DEVICE)

        self.criterion_ = nn.BCEWithLogitsLoss()
        self.optimizer_ = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        y_np = np.asarray(y)
        self.classes_ = np.unique(y_np)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ != 2 and self.num_classes == 1 : # Check consistency for binary case
             print(f"Warning: Binary setup (num_classes=1) but found {self.n_classes_} classes in y.")

        train_dataset = HeartDataset(features_all=X, labels_all=y_np, edge_index=self.edge_index)
        train_loader = PyGDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        if self.verbose > 0:
            print(f"Starting fit: hidden={self.hidden_channels}, lr={self.lr}, epochs={self.epochs}, batch={self.batch_size}")

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_data in train_loader:
                batch_data = batch_data.to(DEVICE)
                self.optimizer_.zero_grad()
                outputs = self.model_(batch_data)
                target = batch_data.y.unsqueeze(1).float()
                loss = self.criterion_(outputs, target)
                loss.backward()
                self.optimizer_.step()
                epoch_loss += loss.item() * batch_data.num_graphs

            avg_epoch_loss = epoch_loss / len(train_loader.dataset)
            if self.verbose > 1 and (epoch + 1) % 10 == 0: # Print progress if verbose
                 print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}")

        if self.verbose > 0:
            print("Fit finished.")
        return self

    def _predict_common(self, X):
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        self.model_.eval()
        X_np = np.asarray(X)
        dummy_labels = np.zeros(len(X_np))
        pred_dataset = HeartDataset(features_all=X_np, labels_all=dummy_labels, edge_index=self.edge_index)
        # Use self.batch_size for prediction loader as well
        pred_loader = PyGDataLoader(pred_dataset, batch_size=self.batch_size, shuffle=False)

        all_outputs = []
        with torch.no_grad():
            for batch_data in pred_loader:
                batch_data = batch_data.to(DEVICE)
                outputs = self.model_(batch_data)
                all_outputs.append(outputs.cpu())
        logits_tensor = torch.cat(all_outputs, dim=0)
        return logits_tensor

    def predict(self, X):
        logits_tensor = self._predict_common(X)
        probs = torch.sigmoid(logits_tensor)
        preds = (probs > 0.5).float().squeeze().numpy()
        return preds.astype(int)

    def predict_proba(self, X):
        logits_tensor = self._predict_common(X)
        prob_class_1 = torch.sigmoid(logits_tensor).numpy()
        prob_class_0 = 1.0 - prob_class_1
        probs = np.hstack((prob_class_0, prob_class_1))
        return probs