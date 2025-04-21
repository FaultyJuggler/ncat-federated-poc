import os
import json
import time
import logging
import threading
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import pickle  # For in-memory model serialization
import base64
from flask import Flask, jsonify, request, Response
import sys


# Add paths for imports
sys.path.append('/')  # Add root directory to path for Docker container
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add parent dir

try:
    from platform_utils import detect_platform, optimize_model_params
except ImportError:
    print("Error: platform_utils.py not found in path")
    print(f"Current sys.path: {sys.path}")
    print("Looking for file in:")
    for path in ['/platform_utils.py', './platform_utils.py', '../platform_utils.py']:
        print(f"  {path}: {os.path.exists(path)}")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("central_server")

# Initialize Flask app
app = Flask(__name__)

# Global variables for FL process
global_model = None
model_type = 'sgd'  # Default model type
client_models = {}
current_round = 0
total_rounds = 10
min_clients = 3
is_training_complete = False
round_metrics = []
lock = threading.Lock()  # Thread lock for model updates

# Detect platform and get optimized parameters
platform_config = detect_platform()
logger.info(f"Running on detected platform: {platform_config['platform']}")


class PyTorchSGDClassifier(BaseEstimator, ClassifierMixin):
    """PyTorch-based SGD classifier with scikit-learn compatible API."""

    def __init__(self, loss='log_loss', penalty='l2', alpha=0.0001,
                 max_iter=1000, tol=1e-3, random_state=42,
                 learning_rate=0.01, batch_size=1000):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Set GPU device if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"PyTorch using device: {self.device}")

        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scaler = StandardScaler()
        self.classes_ = None
        self.n_features_in_ = None
        self.initialized_ = False

        # Set random seed for reproducibility
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

    def _initialize_model(self, n_features, n_classes):
        """Initialize PyTorch model and optimizer."""
        # For binary classification, output dimension should be 1
        n_classes = 1 if len(self.classes_) == 2 else len(self.classes_)

        # Create a simple network with appropriate dimensions
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_features_in_, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_classes)
        ).to(self.device)

        # Use binary cross entropy with class weights for binary classification
        if len(self.classes_) == 2:
            # Calculate class weights if necessary (can also be passed as a parameter)
            # For imbalanced data, try setting weight for minority class higher
            pos_weight = torch.tensor([3.0]).to(self.device)  # Adjust based on your class ratio
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # Setup optimizer with regularization
        weight_decay = self.alpha if self.penalty == 'l2' else 0
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay
        )

        # Set loss function
        if self.loss == 'log_loss' or self.loss == 'log':
            self.criterion = torch.nn.BCEWithLogitsLoss() if out_features == 1 else torch.nn.CrossEntropyLoss()
        elif self.loss == 'hinge':
            self.criterion = torch.nn.MultiMarginLoss()
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

        self.initialized_ = True

    def _prepare_data(self, X, y):
        # Convert to numpy if not already
        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize scaler if not done
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()

        # Fit scaler on first call
        if not hasattr(self, 'scaler_fitted_') or not self.scaler_fitted_:
            X = self.scaler.fit_transform(X)
            self.scaler_fitted_ = True
        else:
            X = self.scaler.transform(X)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Handle target encoding for different loss functions
        if len(self.classes_) == 2:  # Binary classification
            y_tensor = torch.FloatTensor(y).to(self.device)
        else:  # Multi-class classification
            y_tensor = torch.LongTensor(y).to(self.device)

        return X_tensor, y_tensor

    def fit(self, X, y):
        """Fit model to the data."""
        # Check and validate input data
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        # Scale features
        X = self.scaler.fit_transform(X)

        # Convert class labels to indices for multiclass
        if len(self.classes_) > 2:
            y_mapped = np.searchsorted(self.classes_, y)
        else:
            # For binary classification, convert to 0/1
            y_mapped = (y == self.classes_[1]).astype(np.float32)

        # Initialize the PyTorch model if not already done
        if not self.initialized_:
            self._initialize_model(self.n_features_in_, len(self.classes_))

        # Convert to PyTorch tensors
        X_tensor, y_tensor = self._prepare_data(X, y_mapped)

        # Training loop
        self.model.train()
        n_samples = X.shape[0]

        for epoch in range(self.max_iter):
            total_loss = 0
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_tensor[i:i + self.batch_size]
                batch_y = y_tensor[i:i + self.batch_size]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)

                # Reshape output for binary classification
                if len(self.classes_) <= 2:
                    outputs = outputs.squeeze()

                # Compute loss and backward pass
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            # Check convergence
            avg_loss = total_loss / (n_samples / self.batch_size)
            if epoch > 0 and avg_loss < self.tol:
                logger.info(f"Converged after {epoch + 1} epochs. Loss: {avg_loss:.6f}")
                break

            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch + 1}/{self.max_iter}, Loss: {avg_loss:.6f}")

        return self

    def partial_fit(self, X, y, classes=None):
        # Ensure classes are provided for first call or convert from numpy array
        if classes is not None:
            self.classes_ = np.array(classes)
        elif not hasattr(self, 'classes_'):
            self.classes_ = np.unique(y)

        # Set feature dimension on first call
        if not hasattr(self, 'n_features_in_'):
            self.n_features_in_ = X.shape[1]

        # Initialize model if not already done
        if not hasattr(self, 'model') or self.model is None:
            self._initialize_model()

        # Initialize optimizer if not already done
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.alpha
            )

        # Log key info about the data
        logger.info(f"Partial fit with X shape: {X.shape}, y shape: {y.shape}, "
                    f"classes: {self.classes_}, class distribution: {np.bincount(y)}")

        # Prepare batches
        X_tensor, y_tensor = self._prepare_data(X, y)

        # Training loop
        self.model.train()

        # Process in batches to avoid memory issues
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Track metrics
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in dataloader:
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)

            # Handle output shape based on classification type
            if len(self.classes_) == 2:  # Binary classification
                outputs = outputs.view(-1)  # Ensure outputs are flattened for BCE loss
                loss = self.criterion(outputs, batch_y)
            else:  # Multi-class
                loss = self.criterion(outputs, batch_y.long())

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()

            # Calculate batch accuracy
            if len(self.classes_) == 2:
                predicted = (outputs >= 0).float()  # Apply sigmoid implicitly with threshold at 0
            else:
                _, predicted = torch.max(outputs.data, 1)

            batch_correct = (predicted == batch_y).sum().item()
            correct += batch_correct
            total += batch_y.size(0)

        # Log training progress
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total if total > 0 else 0
        logger.info(f"Training - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        return self

    def predict_proba(self, X):
        """Return probability estimates for samples."""
        check_is_fitted(self, ['classes_', 'initialized_'])
        X = check_array(X)
        X = self.scaler.transform(X)

        # Convert to PyTorch tensor
        X_tensor = self._prepare_data(X)

        # Prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

            if len(self.classes_) <= 2:  # Binary classification
                outputs = outputs.squeeze()
                proba = torch.sigmoid(outputs).cpu().numpy()
                return np.vstack((1 - proba, proba)).T
            else:  # Multiclass
                proba = torch.softmax(outputs, dim=1).cpu().numpy()
                return proba

    def predict(self, X):
        """Return predicted class labels for samples in X."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'loss': self.loss,
            'penalty': self.penalty,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    def set_params(self, **parameters):
        """Set parameters for this estimator."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Initialize the global model
def initialize_global_model(X_sample=None, y_sample=None):
    global global_model, model_type

    # If no sample data is provided, create placeholder data
    if X_sample is None or y_sample is None:
        X_sample = np.zeros((2, 9))  # 2 samples, 10 features
        y_sample = np.array([0, 1])  # Binary classification placeholder

    # Get unique classes from sample labels
    unique_classes = np.unique(y_sample)
    n_features = X_sample.shape[1]
    n_classes = len(unique_classes)

    # Get optimized model configuration
    model_config = optimize_model_params(platform_config)

    # Set the default model type to SGD
    model_type = 'sgd'  # Default to SGD classifier

    # Create a helper function for SGD initialization
    def create_sgd_model(config=None):
        from sklearn.linear_model import SGDClassifier

        # Default SGD parameters if not provided
        sgd_params = {
            'loss': 'log_loss',  # Log loss for probability outputs
            'alpha': 0.0001,  # Regularization strength
            'max_iter': 1000,  # Increased from 5 to ensure convergence
            'tol': 1e-3,  # Convergence tolerance
            'random_state': 42,  # For reproducibility
            'warm_start': True,  # Allow incremental training
            'early_stopping': True,  # Enable early stopping
            'validation_fraction': 0.1,  # Use 10% of data for early stopping
            'n_iter_no_change': 5,  # Number of iterations with no improvement
            'learning_rate': 'optimal'  # Automatically adjusts based on regularization
        }

        # Override with any provided config
        if config and 'sgd_params' in config:
            sgd_params.update(config['sgd_params'])

        logger.info(f"Initializing SGDClassifier with parameters: {sgd_params}")
        return SGDClassifier(**sgd_params)

    # Check if the model type in config is explicitly randomforest
    if model_config.get('model_type') == 'randomforest':
        try:
            from sklearn.ensemble import RandomForestClassifier
            global_model = RandomForestClassifier(**model_config['params'])
            model_type = 'randomforest'
            logger.info("Initialized RandomForest model")

            # Also log a warning that SGD is preferred
            logger.warning("Note: SGDClassifier is recommended for federated learning scenarios")
        except Exception:
            global_model = create_sgd_model(model_config)
            logger.info("RandomForest initialization failed. Falling back to SGDClassifier model")

    # Default to SGD model (preferred)
    else:
        try:
            # global_model = create_sgd_model(model_config)
            global_model = PyTorchSGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.001,  # Increased regularization
                max_iter=200,  # More iterations
                tol=1e-4,  # Lower tolerance for better convergence
                random_state=42,
                learning_rate=0.005,  # Lower learning rate for stability
                batch_size=64  # Larger batch size for better convergence
            )

            # Log model info
            logger.info(f"Initializing PyTorch model with X shape: {X_sample.shape}, y shape: {y_sample.shape}")
            logger.info(f"Features: {n_features}, Classes: {n_classes}, class balance: {np.bincount(y_sample)}")

            # Adapt to your PyTorchSGDClassifier implementation
            try:
                # First try: initialize directly with the parameters needed
                if hasattr(global_model, '_initialize_model') and callable(getattr(global_model, '_initialize_model')):
                    # Set attributes directly before calling initialize
                    global_model.n_features_in_ = n_features
                    global_model.classes_ = unique_classes

                    # Try to call _initialize_model with required parameters
                    global_model._initialize_model(n_features=n_features, n_classes=n_classes)
                    logger.info("Initialized PyTorch model directly with parameters")

                # Then call partial_fit with sample data
                global_model.partial_fit(X_sample, y_sample, classes=unique_classes)
                logger.info("Successfully initialized PyTorchSGDClassifier model")
            except TypeError as e:
                logger.error(f"Error initializing PyTorchSGDClassifier: {e}")
                # Fall back to sklearn's SGDClassifier
                global_model = create_sgd_model(model_config)
                global_model.partial_fit(X_sample, y_sample, classes=unique_classes)
                logger.info("Falling back to sklearn SGDClassifier model")
        except Exception as e:
            logger.error(f"PyTorch model initialization failed: {e}")
            # Fall back to sklearn's SGDClassifier
            global_model = create_sgd_model(model_config)
            global_model.partial_fit(X_sample, y_sample, classes=unique_classes)
            logger.info("Falling back to sklearn SGDClassifier model")

    return True


def serialize_model(model):
    """
    Serialize a model (scikit-learn or PyTorchSGDClassifier) to a JSON-serializable format.
    """
    if model is None:
        return {'model_present': False}

    serialized = {'model_present': True}

    # Add model type information
    serialized['model_type'] = model.__class__.__name__

    # Add shape information if available
    if hasattr(model, 'n_features_in_'):
        serialized['n_features'] = int(model.n_features_in_)

    # Safely handle classes attribute - check both existence and non-None value
    if hasattr(model, 'classes_') and model.classes_ is not None:
        serialized['classes'] = model.classes_.tolist()
    else:
        serialized['classes'] = None

    # Add sklearn-specific attributes
    if hasattr(model, 'coef_') and model.coef_ is not None:
        serialized['coef'] = model.coef_.tolist()
    if hasattr(model, 'intercept_') and model.intercept_ is not None:
        serialized['intercept'] = model.intercept_.tolist()

    # Special handling for PyTorchSGDClassifier
    if model.__class__.__name__ == 'PyTorchSGDClassifier':
        # Serialize hyperparameters
        if hasattr(model, 'get_params'):
            serialized['params'] = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                                    for k, v in model.get_params().items()}

        # Serialize the PyTorch model state
        if hasattr(model, 'model') and model.model is not None:
            # Convert PyTorch model state to bytes and then base64 string
            try:
                state_buffer = io.BytesIO()
                torch.save(model.model.state_dict(), state_buffer)
                serialized['state_dict'] = base64.b64encode(state_buffer.getvalue()).decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to serialize PyTorch model state: {e}")

    # For sklearn models, add other parameters
    elif hasattr(model, 'get_params'):
        serialized['params'] = {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
                                for k, v in model.get_params().items()}

    return serialized



def deserialize_model(serialized_params):
    """
    Deserialize model parameters into a model object.
    Uses PyTorch for GPU acceleration when available.
    """
    logger.info("Deserializing model...")

    # Get model type, default to 'sgd'
    model_type = serialized_params.get('model_type', 'sgd')
    use_gpu = serialized_params.get('use_gpu', platform_config.get('use_gpu', False))

    if model_type == 'sgd':
        try:
            # Try to use PyTorch if GPU acceleration is requested
            if use_gpu:
                try:
                        # Extract SGD-specific parameters
                    sgd_params = {
                        'loss': serialized_params.get('loss', 'log_loss'),
                        'penalty': serialized_params.get('penalty', 'l2'),
                        'alpha': serialized_params.get('alpha', 0.0001),
                        'max_iter': serialized_params.get('max_iter', 1000),
                        'tol': serialized_params.get('tol', 1e-3),
                        'random_state': serialized_params.get('random_state', 42),
                        'learning_rate': serialized_params.get('learning_rate', 0.01),
                        'batch_size': serialized_params.get('batch_size', 1000)
                    }

                    logger.info("Creating PyTorch GPU-accelerated SGD classifier")
                    return PyTorchSGDClassifier(**sgd_params)

                except ImportError:
                    logger.warning("GPU acceleration requested but PyTorch not available. Falling back to CPU SGD.")

            # CPU-based scikit-learn SGD
            from sklearn.linear_model import SGDClassifier

            # Extract SGD-specific parameters for scikit-learn
            sgd_params = {
                'loss': serialized_params.get('loss', 'log_loss'),  # Updated from 'log' for newer sklearn
                'penalty': serialized_params.get('penalty', 'l2'),
                'alpha': serialized_params.get('alpha', 0.0001),
                'max_iter': serialized_params.get('max_iter', 1000),
                'tol': serialized_params.get('tol', 1e-3),
                'random_state': serialized_params.get('random_state', 42),
                'warm_start': True  # Enables incremental learning across model updates
            }

            model = PyTorchSGDClassifier(BaseEstimator, ClassifierMixin)

            logger.info("Creating CPU SGD classifier")
            return SGDClassifier(**sgd_params)

        except Exception as e:
            logger.error(f"Error creating SGD model: {e}")
            raise
    else:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**serialized_params['params'])

    return model


def merge_models(models, sample_counts):
    """
    Merge models using weighted voting based on sample counts.
    Handles different model types appropriately.
    """
    # Basic validation
    if not models:
        raise ValueError("No models provided for merging")

    if not all(type(models[0]) == type(m) for m in models):
        logger.warning("Inconsistent model types detected. Converting to same type.")

    # Get model type from first model
    model_type = 'sgd'  # Default to SGD

    # Check if the model is SGDClassifier (recommended approach)
    if hasattr(models[0], '_finalize_coef'):  # SGDClassifier has this internal method
        logger.info("Merging SGD Classifier models")
        model_type = 'sgd'
    # Fallback checks for other model types
    elif hasattr(models[0], 'estimators_'):
        logger.warning("RandomForest model detected, but SGD is recommended")
        model_type = 'randomforest'

    # Create a new global model with optimized parameters for current platform
    model_config = optimize_model_params(platform_config)

    # Override model_type to ensure SGD is used
    model_config['model_type'] = 'sgd'

    if model_type == 'sgd':
        # For SGDClassifier
        from sklearn.linear_model import SGDClassifier
        global_model = SGDClassifier(**model_config['params'])

        # Merge SGD models by averaging coefficients and intercepts
        if models and all(hasattr(m, 'coef_') for m in models):
            # Calculate weights based on sample counts
            total_samples = sum(sample_counts)
            weights = [count / total_samples for count in sample_counts]

            # Get first model's shapes to initialize
            reference_model = models[0]
            merged_coef = np.zeros_like(reference_model.coef_)
            merged_intercept = np.zeros_like(reference_model.intercept_)

            # Weighted average of coefficients and intercepts
            for model, weight in zip(models, weights):
                merged_coef += model.coef_ * weight
                merged_intercept += model.intercept_ * weight

            # Set the averaged parameters to the global model
            global_model.coef_ = merged_coef
            global_model.intercept_ = merged_intercept

            # Copy other necessary attributes from reference model
            global_model.classes_ = reference_model.classes_
            global_model.n_features_in_ = reference_model.n_features_in_

            # Set the model as fitted
            global_model._fitted = True
        else:
            logger.warning("Some SGD models don't have coefficients. Using model with most data.")
            global_model = models[np.argmax(sample_counts)]

    else:
        # For RandomForest (fallback case)
        logger.warning("Using RandomForest model instead of recommended SGD Classifier")
        from sklearn.ensemble import RandomForestClassifier
        global_model = RandomForestClassifier(**model_config['params'])

        # Calculate the total number of trees and weights per client
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]

        # Calculate how many trees to take from each client model
        total_trees = global_model.n_estimators
        trees_per_model = [int(round(weight * total_trees)) for weight in weights]

        # Adjust to ensure we get exactly total_trees
        while sum(trees_per_model) < total_trees:
            idx = trees_per_model.index(min(trees_per_model))
            trees_per_model[idx] += 1
        while sum(trees_per_model) > total_trees:
            idx = trees_per_model.index(max(trees_per_model))
            trees_per_model[idx] -= 1

        # Gather trees from each model according to their weights
        merged_estimators = []
        for model, n_trees in zip(models, trees_per_model):
            if n_trees <= 0:
                continue
            # For simplicity, take the first n_trees
            merged_estimators.extend(model.estimators_[:n_trees])

        # Update the global model's estimators
        global_model.estimators_ = merged_estimators

        # Ensure other attributes are correctly set
        if models:
            reference_model = models[0]
            global_model.n_classes_ = reference_model.n_classes_
            global_model.classes_ = reference_model.classes_
            global_model.n_features_in_ = reference_model.n_features_in_

    return global_model


def federated_averaging():
    global global_model, current_round, client_models

    logger.info(f"Performing federated averaging for round {current_round}")

    if len(client_models) == 0:
        logger.warning("No client models available for aggregation")
        return False

    # Collect models and their weights (sample counts)
    models = []
    weights = []

    # Use the in-memory models directly from client_models dictionary
    for client_id, client_data in client_models.items():
        if 'model' in client_data and client_data['model'] is not None:
            models.append(client_data['model'])
            weights.append(client_data['sample_count'])
            logger.info(f"Using in-memory model from client {client_id} with weight {client_data['sample_count']}")
        else:
            logger.warning(f"No valid model found for client {client_id}")

    if len(models) == 0:
        logger.warning("No valid models to aggregate")
        return False

    try:
        # Normalize weights to sum to 1.0
        total_samples = sum(weights)
        weights = [w / total_samples for w in weights]

        # Check if model has coefficient and intercept attributes (like sklearn models)
        if hasattr(models[0], 'coef_') and hasattr(models[0], 'intercept_'):
            # Create a new model with the same parameters
            aggregated_model = clone(models[0])

            # Average the coefficients
            coefs = np.array([model.coef_ for model in models])
            aggregated_model.coef_ = np.sum([coef * weight for coef, weight in zip(coefs, weights)], axis=0)

            # Average the intercepts
            intercepts = np.array([model.intercept_ for model in models])
            aggregated_model.intercept_ = np.sum([intercept * weight for intercept, weight in zip(intercepts, weights)],
                                                 axis=0)

            global_model = aggregated_model
            logger.info("Successfully created aggregated model by averaging coefficients and intercepts")
        else:
            # For custom model types, use your existing merge_models function
            logger.info("Models don't have standard coef_/intercept_ attributes, using custom merge function")
            global_model = merge_models(models, weights)

        # Make sure the global model is valid
        if global_model is not None:
            # Define data directory and ensure it exists
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            os.makedirs(data_dir, exist_ok=True)

            # Save the model
            model_path = os.path.join(data_dir, f"global_model_round_{current_round}.joblib")
            joblib.dump(global_model, model_path)
            logger.info(f"Saved aggregated global model for round {current_round} to {model_path}")
            return True
        else:
            logger.error("Failed to create a valid global model")
            return False

    except Exception as e:
        logger.error(f"Error during federated averaging: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False



# API endpoints
@app.route('/status', methods=['GET'])
def get_status():
    """Return the current status of the federated learning process"""
    return jsonify({
        "status": "completed" if is_training_complete else "in_progress",
        "current_round": current_round,
        "total_rounds": total_rounds,
        "connected_clients": list(client_models.keys()),
        "platform": platform_config['platform'],
        "model_type": model_type,
        "gpu_enabled": platform_config['use_gpu']
    })


@app.route('/model', methods=['GET'])
def get_model():
    """Return the current global model serialized representation"""
    return jsonify(serialize_model(global_model))


@app.route('/upload', methods=['POST'])
def upload_model():
    global current_round, client_models, is_training_complete

    if request.content_type and 'application/json' in request.content_type:
        data = request.json
        client_id = data['client_id']
        sample_count = data['sample_count']
        metrics = data.get('metrics', {})
        model_encoded = data.get('model')

        # Decode and deserialize the model
        if model_encoded:
            model_bytes = base64.b64decode(model_encoded)
            model = pickle.loads(model_bytes)  # Deserialize back to the model object
            logger.info(f"Successfully deserialized model for client {client_id}")

        # Save metadata in memory
        with lock:
            client_models[client_id] = {
                'sample_count': sample_count,
                'metrics': metrics,
                'model': model,  # Store the deserialized model (optional)
                'timestamp': time.time()
            }

        # Check if all expected clients have submitted their models
        logger.info(f"Received model from client {client_id}. Models received: {len(client_models)}/{min_clients}")

        if len(client_models) >= min_clients:
            try:
                # Aggregate models using federated averaging
                logger.info(f"All {min_clients} clients submitted models. Performing federated averaging...")
                if federated_averaging():
                    # Create metrics for this round
                    metrics = {
                        'participating_clients': len(client_models),
                        'total_samples': sum(
                            client_data['sample_count'] for client_id, client_data in client_models.items()),
                        # Add any other metrics you want to track
                    }

                    # Append to round_metrics list instead of using as a dictionary
                    round_metrics.append(metrics)

                    # Advance to next round
                    current_round += 1
                    # Reset client models for the new round
                    client_models = {}

                    logger.info(f"Successfully completed round {current_round - 1}, advancing to round {current_round}")
                else:
                    logger.error("Federated averaging failed, not advancing to next round")

                # Check if training is complete
                if current_round >= total_rounds:
                    is_training_complete = True
                    logger.info("Training complete!")
            except Exception as e:
                logger.error(f"Failed to advance round: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

    return jsonify({
            "status": "success",
            "message": f"Model received from client {client_id}",
            "current_round": current_round,
        }), 200


@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Return training metrics for all rounds"""
    return jsonify(round_metrics)

@app.route('/platform', methods=['GET'])
def get_platform_info():
    """Return information about the platform this server is running on"""
    return jsonify(platform_config)


if __name__ == "__main__":
    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Initialize the global model
    initialize_global_model()

    logger.info(f"Starting federated learning server with model type: {model_type}")
    logger.info(f"Platform: {platform_config['platform']}, GPU enabled: {platform_config['use_gpu']}")
    app.run(host='0.0.0.0', port=8080)