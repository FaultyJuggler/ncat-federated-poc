import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logger = logging.getLogger(__name__)


# Create a PyTorch-based SGD classifier wrapper
class PyTorchSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, loss='log', penalty='l2', alpha=0.0001,
                 max_iter=1000, tol=1e-3, random_state=42,
                 learning_rate=0.01, batch_size=32):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scaler = StandardScaler()
        self.scaler_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        self.initialized_ = False

    def _initialize_model(self, n_features, n_classes):
        """Initialize the PyTorch model with the correct dimensions"""
        logger.info(f"Initializing PyTorch model with {n_features} features and {n_classes} classes")

        # Create a simple neural network model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_features, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, n_classes if n_classes > 2 else 1)
        ).to(self.device)

        # Set loss function based on number of classes
        self.criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()

        # Set optimizer with appropriate L2 regularization
        weight_decay = self.alpha if self.penalty == 'l2' else 0.0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay
        )

        self.initialized_ = True

    def _prepare_data(self, X, y=None):
        """Prepare data for PyTorch model"""
        # Store number of features
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        # Apply scaling
        if not self.scaler_fitted_ and y is not None:
            self.scaler.fit(X)
            self.scaler_fitted_ = True
        X_scaled = self.scaler.transform(X) if self.scaler_fitted_ else X

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        if y is not None:
            y_tensor = torch.LongTensor(y).to(self.device)
            if len(self.classes_) == 2:
                y_tensor = y_tensor.float()
            return X_tensor, y_tensor

        return X_tensor

    def fit(self, X, y):
        """Fit the model to the data"""
        # Convert y to numpy array if it's not
        y = np.asarray(y)

        # Store unique classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Map classes to indices if necessary
        y_indices = np.searchsorted(self.classes_, y)

        # Initialize model if not already done
        if not self.initialized_:
            self._initialize_model(X.shape[1], n_classes)

        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y_indices)

        # Training loop
        self.model.train()
        prev_loss = float('inf')

        for epoch in range(self.max_iter):
            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_tensor)

            # Compute loss
            if len(self.classes_) == 2:
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, y_tensor)
            else:
                loss = self.criterion(outputs, y_tensor)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Check for convergence
            current_loss = loss.item()
            if epoch > 0 and abs(prev_loss - current_loss) < self.tol:
                logger.info(f"Converged at epoch {epoch} with loss {current_loss:.6f}")
                break

            prev_loss = current_loss

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {current_loss:.6f}")

        return self

    def partial_fit(self, X, y, classes=None):
        """Partially fit the model to the data (for online learning)"""
        # Set or update classes if provided
        if classes is not None:
            self.classes_ = np.asarray(classes)
        elif self.classes_ is None:
            self.classes_ = np.unique(y)

        n_classes = len(self.classes_)

        # Map classes to indices
        y_indices = np.searchsorted(self.classes_, y)

        # Initialize model if not already done
        if not self.initialized_:
            self._initialize_model(X.shape[1], n_classes)

        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y_indices)

        # Training
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(X_tensor)

        # Compute loss
        if len(self.classes_) == 2:
            outputs = outputs.squeeze()
            loss = self.criterion(outputs, y_tensor)
        else:
            loss = self.criterion(outputs, y_tensor)

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        return self

    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.initialized_:
            raise ValueError("Model not initialized. Call fit or partial_fit first.")

        # Prepare data
        X_tensor = self._prepare_data(X)

        # Set model to evaluation mode
        self.model.eval()

        # Get predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)

            # For binary classification
            if len(self.classes_) == 2:
                outputs = outputs.squeeze()
                proba = torch.sigmoid(outputs).cpu().numpy()
                return np.vstack([1 - proba, proba]).T

            # For multiclass classification
            else:
                proba = F.softmax(outputs, dim=1).cpu().numpy()
                return proba

    def predict(self, X):
        """Predict classes"""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        params = {
            'loss': self.loss,
            'penalty': self.penalty,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        return params

    def set_params(self, **parameters):
        """Set parameters for this estimator"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class SimpleNN(nn.Module):
    """Simple neural network for federated learning example."""

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten for MNIST-like data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# def train(model, train_loader, epochs, device="cpu"):
#     """Train the model on the training set."""
#     model.train()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#     criterion = nn.CrossEntropyLoss()
#
#     for epoch in range(epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
#
#         for data, target in train_loader:
#             data, target = data.to(device), target.to(device)
#
#             # Zero the parameter gradients
#             optimizer.zero_grad()
#
#             # Forward + backward + optimize
#             outputs = model(data)
#             loss = criterion(outputs, target)
#             loss.backward()
#             optimizer.step()
#
#             # Statistics
#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#
#         # Print epoch statistics
#         accuracy = 100 * correct / total
#         print(f'Epoch {epoch + 1}: Loss: {running_loss / len(train_loader):.3f}, '
#               f'Accuracy: {accuracy:.2f}%')
#
#     return model
#
#
# def evaluate(model, test_loader, device="cpu"):
#     """Evaluate the model on the test set."""
#     model.eval()
#     criterion = nn.CrossEntropyLoss()
#     test_loss = 0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             outputs = model(data)
#             test_loss += criterion(outputs, target).item()
#             _, predicted = torch.max(outputs.data, 1)
#             total += target.size(0)
#             correct += (predicted == target).sum().item()
#
#     accuracy = 100 * correct / total
#     avg_loss = test_loss / len(test_loader)
#     print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
#
#     return {"loss": avg_loss, "accuracy": accuracy}