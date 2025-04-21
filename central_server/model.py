# model.py
import numpy as np
import base64
import logging
import pickle
from io import BytesIO
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger('central_server')

# Configure PyTorch to use deterministic algorithms for reproducibility
torch.use_deterministic_algorithms(True, warn_only=True)

class PyTorchSGDClassifier(BaseEstimator, ClassifierMixin):
    """
    A PyTorch implementation of SGD Classifier with similar API to sklearn's SGDClassifier
    """

    def __init__(self, loss='log_loss', penalty='l2', alpha=0.0001, max_iter=1000,
                 tol=1e-3, random_state=None, learning_rate=0.01, batch_size=32):
        """
        Initialize the PyTorch SGD Classifier

        Args:
            loss: Loss function ('log_loss' for logistic regression)
            penalty: Regularization penalty ('l2' or None)
            alpha: Regularization strength
            max_iter: Maximum number of epochs
            tol: Stopping criterion tolerance
            random_state: Random seed for reproducibility
            learning_rate: Learning rate for SGD
            batch_size: Mini-batch size for training
        """
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Set random seed for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Initialize device (CPU/GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Will be initialized when fit is called
        self.model = None
        self.optimizer = None
        self.scaler = StandardScaler()
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
        """
        Scale features and convert to PyTorch tensors

        Args:
            X: Input features
            y: Target labels (optional)

        Returns:
            X_tensor: PyTorch tensor of features
            y_tensor: PyTorch tensor of labels (or None if y was None)
        """
        # Apply scaling only during fit to avoid data leakage
        if y is not None and not hasattr(self, 'scaler_fitted_'):
            X = self.scaler.fit_transform(X)
            self.scaler_fitted_ = True
        elif hasattr(self, 'scaler_fitted_'):
            X = self.scaler.transform(X)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Convert labels if provided
        if y is not None:
            # For binary classification, we need labels as floats
            if len(np.unique(y)) == 2:
                y_tensor = torch.FloatTensor(y).to(self.device)
            else:
                y_tensor = torch.LongTensor(y).to(self.device)
            return X_tensor, y_tensor

        return X_tensor

    def fit(self, X, y):
        """
        Fit the model to data

        Args:
            X: Features
            y: Target labels

        Returns:
            self: The fitted estimator
        """
        # Extract unique classes and store them
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Map classes to integers if they aren't already
        if not np.all(np.unique(y) == np.arange(n_classes)):
            y_mapped = np.zeros(y.shape, dtype=int)
            for i, cls in enumerate(self.classes_):
                y_mapped[y == cls] = i
            y = y_mapped

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Initialize the model if not already done
        if not self.initialized_:
            self._initialize_model(n_features, n_classes)

        # Convert data to PyTorch tensors with scaling
        X_tensor, y_tensor = self._prepare_data(X, y)

        # Training loop
        self.model.train()
        best_loss = float('inf')
        no_improvement = 0

        for epoch in range(self.max_iter):
            epoch_loss = 0.0
            n_batches = 0

            # Create batches
            indices = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), self.batch_size):
                # Get batch
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_tensor[batch_indices]
                y_batch = y_tensor[batch_indices]

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)

                # Reshape outputs for loss calculation
                if n_classes == 2:
                    outputs = outputs.view(-1)

                # Compute loss
                loss = self.criterion(outputs, y_batch)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Calculate average loss for the epoch
            avg_loss = epoch_loss / n_batches

            # Early stopping based on training loss
            if avg_loss < best_loss - self.tol:
                best_loss = avg_loss
                no_improvement = 0
            else:
                no_improvement += 1

            # If no improvement for several epochs, stop training
            if no_improvement >= 5:  # Patience parameter
                logger.info(f"Early stopping at epoch {epoch}")
                break

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Incrementally fit the model to batches of data

        Args:
            X: Features
            y: Target labels
            classes: Array of all possible classes

        Returns:
            self: The fitted estimator
        """
        # Check if the model is initialized
        if not self.initialized_:
            # Store classes if provided, otherwise extract from y
            if classes is not None:
                self.classes_ = classes
            else:
                self.classes_ = np.unique(y)

            n_classes = len(self.classes_)
            n_samples, n_features = X.shape
            self.n_features_in_ = n_features

            self._initialize_model(n_features, n_classes)

        # Map classes to integers if they aren't already
        n_classes = len(self.classes_)
        if not np.all(np.unique(y) == np.arange(n_classes)):
            y_mapped = np.zeros(y.shape, dtype=int)
            for i, cls in enumerate(self.classes_):
                y_mapped[y == cls] = i
            y = y_mapped

        # Convert data to PyTorch tensors with scaling
        X_tensor, y_tensor = self._prepare_data(X, y)

        # Train for one epoch
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Create batches
        indices = torch.randperm(len(X_tensor))
        for i in range(0, len(X_tensor), self.batch_size):
            # Get batch
            batch_indices = indices[i:i + self.batch_size]
            X_batch = X_tensor[batch_indices]
            y_batch = y_tensor[batch_indices]

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)

            # Reshape outputs for loss calculation
            if n_classes == 2:
                outputs = outputs.view(-1)

            # Compute loss
            loss = self.criterion(outputs, y_batch)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Calculate average loss
        avg_loss = epoch_loss / max(1, n_batches)
        logger.debug(f"Partial fit batch loss: {avg_loss:.6f}")

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples

        Args:
            X: Features

        Returns:
            proba: Class probabilities
        """
        if not self.initialized_:
            raise RuntimeError("Model is not initialized. Call fit or partial_fit first.")

        # Convert data to PyTorch tensors with scaling
        X_tensor = self._prepare_data(X)

        # Set to evaluation mode
        self.model.eval()

        # Make predictions
        with torch.no_grad():
            outputs = self.model(X_tensor)

            # Convert outputs to probabilities
            if len(self.classes_) == 2:
                # For binary classification
                outputs = outputs.view(-1)
                probas = torch.sigmoid(outputs).cpu().numpy()
                return np.vstack([1 - probas, probas]).T
            else:
                # For multi-class
                return torch.softmax(outputs, dim=1).cpu().numpy()

    def predict(self, X):
        """
        Predict class labels for input samples

        Args:
            X: Features

        Returns:
            y_pred: Predicted class labels
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
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
        """Set parameters for this estimator"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


def create_global_model(params=None):
    """
    Create a new RandomForest model with provided parameters or defaults

    Args:
        params: Dictionary of parameters for RandomForestClassifier

    Returns:
        RandomForestClassifier: Initialized model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'random_state': 42
        }

    return RandomForestClassifier(**params)


def optimize_model_params(config):
    """Optimize model configuration based on platform settings"""
    # Extract relevant config parameters
    model_config = {}

    # Set defaults
    model_config['model_type'] = 'pytorch_sgd'
    model_config['sgd_params'] = {
        'loss': 'log_loss',
        'alpha': 0.0001,
        'max_iter': 1000,
        'random_state': 42
    }

    # Override with config if provided
    if config and 'model' in config:
        if 'type' in config['model']:
            model_config['model_type'] = config['model']['type']
        if 'params' in config['model']:
            model_config['params'] = config['model']['params']

    return model_config


def initialize_global_model(X_sample=None, y_sample=None):
    """Initialize the global model based on sample data or synthetic data

    Args:
        X_sample: Optional sample features
        y_sample: Optional sample labels

    Returns:
        model: Initialized global model
        model_type: Type of model initialized
    """
    # If no sample data is provided, create more representative placeholder data
    if X_sample is None or y_sample is None:
        # Create a larger sample that better represents your data scale
        n_features = 9  # Adjust based on your actual feature count
        n_samples = 1000  # Larger sample for better initialization

        # Create synthetic features
        X_sample = np.random.randn(n_samples, n_features)

        # Create labels with realistic class imbalance
        y_sample = np.zeros(n_samples, dtype=int)
        y_sample[np.random.choice(n_samples, int(n_samples * 0.25), replace=False)] = 1

        logger.info(f"Created synthetic initialization data with {n_samples} samples")
        logger.info(f"Synthetic class distribution: {np.bincount(y_sample)}")
    else:
        logger.info(f"Using provided sample data with {len(y_sample)} samples")
        logger.info(f"Sample class distribution: {np.bincount(y_sample)}")

    # Get unique classes from sample labels
    unique_classes = np.unique(y_sample)
    n_features = X_sample.shape[1]
    n_classes = len(unique_classes)

    # Log important dimensions
    logger.info(f"Features: {n_features}, Classes: {n_classes}")

    # Get optimized model configuration
    model_config = optimize_model_params(None)  # Pass config if available

    # Define a function to create an sklearn SGD model with correct params
    def create_sgd_model(config=None):
        from sklearn.linear_model import SGDClassifier

        # SGD parameters - IMPORTANT: early_stopping must be False for partial_fit
        sgd_params = {
            'loss': 'log_loss',
            'alpha': 0.0001,
            'max_iter': 1000,
            'tol': 1e-3,
            'random_state': 42,
            'warm_start': True,
            'early_stopping': False,  # Must be False for partial_fit
            'learning_rate': 'adaptive',
            'eta0': 0.01
        }

        # Override with any provided config, forcing early_stopping=False
        if config and 'sgd_params' in config:
            config_params = config['sgd_params'].copy()
            config_params['early_stopping'] = False  # Force this to be False
            sgd_params.update(config_params)

        logger.info(f"Initializing SGDClassifier with parameters: {sgd_params}")
        return SGDClassifier(**sgd_params)

    # Try initializing the model based on the config
    try:
        # First try with PyTorch model
        logger.info("Attempting to initialize PyTorch model")

        # Create the PyTorchSGDClassifier
        model = PyTorchSGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=100,
            tol=1e-3,
            random_state=42,
            learning_rate=0.01,
            batch_size=128
        )

        # Set required attributes
        model.n_features_in_ = n_features
        model.classes_ = unique_classes

        # Initialize the model
        model._initialize_model(n_features, n_classes)
        logger.info("PyTorch model initialization successful")

        # Fit the model with a few epochs on the sample data
        for epoch in range(3):
            model.partial_fit(X_sample, y_sample, classes=unique_classes)

        logger.info("PyTorch model training successful")
        model_type = 'pytorch_sgd'

    except Exception as e:
        logger.error(f"PyTorch model initialization failed: {e}")
        logger.info("Falling back to sklearn SGDClassifier")

        # Fallback to sklearn SGD model
        try:
            # Try using RandomForest if specified in config
            if model_config.get('model_type') == 'randomforest':
                try:
                    from sklearn.ensemble import RandomForestClassifier
                    rf_params = model_config.get('params', {'n_estimators': 100, 'random_state': 42})
                    rf_params['class_weight'] = 'balanced'  # Add class balancing
                    model = RandomForestClassifier(**rf_params)
                    model.fit(X_sample, y_sample)  # RandomForest doesn't use partial_fit
                    model_type = 'randomforest'
                    logger.info("RandomForest model initialization successful")
                except Exception as rf_error:
                    logger.error(f"RandomForest initialization failed: {rf_error}")
                    model = create_sgd_model(model_config)
                    model.partial_fit(X_sample, y_sample, classes=unique_classes)
                    model_type = 'sgd'
                    logger.info("SGD model initialization successful as fallback")
            else:
                # Default to SGD
                model = create_sgd_model(model_config)
                model.partial_fit(X_sample, y_sample, classes=unique_classes)
                model_type = 'sgd'
                logger.info("SGD model initialization successful")

        except Exception as sgd_error:
            logger.error(f"SGD model initialization also failed: {sgd_error}")
            raise RuntimeError("Failed to initialize any model type")

    return model, model_type


def serialize_model(model, model_type):
    """
    Serialize model to a format that can be sent over the network

    Args:
        model: The model to serialize
        model_type: Type of model ('pytorch_sgd', 'sgd', 'randomforest')

    Returns:
        dict: Serialized model representation
    """
    try:
        if model_type == 'pytorch_sgd':
            # Serialize PyTorch model
            buffer = BytesIO()

            # Save PyTorch model state dict
            model_state = {
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'scaler': pickle.dumps(model.scaler),
                'classes': model.classes_.tolist(),
                'n_features_in': model.n_features_in_,
                'hyperparams': model.get_params()
            }

            torch.save(model_state, buffer)
            serialized = {
                'model_type': 'pytorch_sgd',
                'model_data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                'classes': model.classes_.tolist()
            }

        elif model_type == 'sgd':
            # Serialize sklearn SGD model
            buffer = BytesIO()
            pickle.dump(model, buffer)
            serialized = {
                'model_type': 'sgd',
                'model_data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                'classes': model.classes_.tolist()
            }

        elif model_type == 'randomforest':
            # Serialize RandomForest model - more detailed to reduce size
            trees = []
            for tree in model.estimators_:
                tree_dict = {
                    'node_count': tree.tree_.node_count,
                    'children_left': tree.tree_.children_left.tolist(),
                    'children_right': tree.tree_.children_right.tolist(),
                    'feature': tree.tree_.feature.tolist(),
                    'threshold': tree.tree_.threshold.tolist(),
                    'values': tree.tree_.value.tolist()
                }
                trees.append(tree_dict)

            buffer = BytesIO()
            pickle.dump({
                'trees': trees,
                'n_classes': model.n_classes_,
                'n_features': model.n_features_in_,
                'classes': model.classes_.tolist(),
                'params': {
                    'n_estimators': model.n_estimators,
                    'criterion': model.criterion,
                    'max_depth': model.max_depth,
                    'min_samples_split': model.min_samples_split,
                    'min_samples_leaf': model.min_samples_leaf,
                    'bootstrap': model.bootstrap
                }
            }, buffer)

            serialized = {
                'model_type': 'randomforest',
                'model_data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
                'classes': model.classes_.tolist()
            }
        else:
            raise ValueError(f"Unsupported model type for serialization: {model_type}")

        return serialized

    except Exception as e:
        logger.error(f"Model serialization failed: {e}")
        raise


def deserialize_model(serialized_model):
    """
    Deserialize a model from its serialized representation

    Args:
        serialized_model: Dictionary containing serialized model

    Returns:
        tuple: (model, model_type)
    """
    try:
        model_type = serialized_model['model_type']
        model_data = base64.b64decode(serialized_model['model_data'])

        if model_type == 'pytorch_sgd':
            # Create empty PyTorch model
            model = PyTorchSGDClassifier()

            # Load model state
            buffer = BytesIO(model_data)
            state = torch.load(buffer, map_location=model.device)

            # Get model parameters
            n_features_in = state['n_features_in']
            classes = np.array(state['classes'])
            n_classes = len(classes)

            # Set model attributes
            model.classes_ = classes
            model.n_features_in_ = n_features_in

            # Initialize the model structure
            model._initialize_model(n_features_in, n_classes)

            # Load state dictionaries
            model.model.load_state_dict(state['model_state_dict'])
            model.optimizer.load_state_dict(state['optimizer_state_dict'])
            model.scaler = pickle.loads(state['scaler'])

            # Set hyperparameters
            model.set_params(**state['hyperparams'])
            model.initialized_ = True

        elif model_type == 'sgd':
            # Deserialize sklearn SGD model
            buffer = BytesIO(model_data)
            model = pickle.load(buffer)

        elif model_type == 'randomforest':
            # Deserialize RandomForest model
            buffer = BytesIO(model_data)
            model_state = pickle.load(buffer)

            # Create new RandomForestClassifier with saved parameters
            model = RandomForestClassifier(**model_state['params'])

            # Set required attributes
            model.n_features_in_ = model_state['n_features']
            model.classes_ = np.array(model_state['classes'])
            model.n_classes_ = model_state['n_classes']

            # More complex reconstruction would be needed for full RF restore
            # This is a simplified approach

        else:
            raise ValueError(f"Unsupported model type for deserialization: {model_type}")

        return model, model_type

    except Exception as e:
        logger.error(f"Model deserialization failed: {e}")
        raise


def merge_models(models, model_types, sample_counts):
    """
    Merge multiple models using federated averaging

    Args:
        models: List of models to merge
        model_types: List of model types corresponding to models
        sample_counts: List of sample counts for each model for weighted averaging

    Returns:
        tuple: (merged_model, model_type)
    """
    # Validate inputs
    if not models or len(models) == 0:
        raise ValueError("No models provided for merging")

    if len(models) != len(sample_counts) or len(models) != len(model_types):
        raise ValueError("Mismatch in number of models, types and sample counts")

    # Check if all models are of the same type
    model_type = model_types[0]
    if not all(mt == model_type for mt in model_types):
        logger.warning("Attempting to merge models of different types. " +
                       "Will use the most common model type.")

        # Use the most common model type
        from collections import Counter
        model_type_counts = Counter(model_types)
        model_type = model_type_counts.most_common(1)[0][0]

        # Filter models to only keep those of the selected type
        filtered_models = []
        filtered_samples = []

        for model, mt, samples in zip(models, model_types, sample_counts):
            if mt == model_type:
                filtered_models.append(model)
                filtered_samples.append(samples)

        models = filtered_models
        sample_counts = filtered_samples

        if not models:
            raise ValueError("No compatible models for merging")

    # Merge based on model type
    if model_type == 'pytorch_sgd':
        # Federated averaging for PyTorch models
        merged_model = federated_averaging(models, sample_counts)
        return merged_model, model_type

    elif model_type == 'sgd':
        # For sklearn SGD, we need to average coefficients
        # Create a new model based on the first model's parameters
        merged_model = SGDClassifier(**models[0].get_params())

        # Set required attributes
        merged_model.classes_ = models[0].classes_
        merged_model.n_features_in_ = models[0].n_features_in_

        # Average coefficients and intercepts with weighted averaging
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]

        # Initialize coefficient and intercept arrays
        coef = np.zeros_like(models[0].coef_)
        intercept = np.zeros_like(models[0].intercept_)

        # Weighted average of coefficients and intercepts
        for model, weight in zip(models, weights):
            coef += model.coef_ * weight
            intercept += model.intercept_ * weight

        # Set coefficients and intercepts in the merged model
        merged_model.coef_ = coef
        merged_model.intercept_ = intercept

        return merged_model, model_type

    elif model_type == 'randomforest':
        # For RandomForest, we use a different strategy
        # Take trees from each model proportionally
        merged_model = create_global_model({
            'n_estimators': models[0].n_estimators,
            'criterion': models[0].criterion,
            'max_depth': models[0].max_depth,
            'min_samples_split': models[0].min_samples_split,
            'min_samples_leaf': models[0].min_samples_leaf,
            'bootstrap': models[0].bootstrap,
            'random_state': 42
        })

        # Calculate the total number of trees and weights per client
        total_samples = sum(sample_counts)
        weights = [count / total_samples for count in sample_counts]

        # Calculate how many trees to take from each client model
        total_trees = merged_model.n_estimators
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
        merged_model.estimators_ = merged_estimators

        # Also need to ensure other attributes are correctly set
        reference_model = models[0]
        merged_model.n_classes_ = reference_model.n_classes_
        merged_model.classes_ = reference_model.classes_
        merged_model.n_features_in_ = reference_model.n_features_in_

        return merged_model, model_type

    else:
        raise ValueError(f"Unsupported model type for merging: {model_type}")


def federated_averaging(models, sample_counts):
    """
    Perform federated averaging on PyTorch models

    Args:
        models: List of PyTorchSGDClassifier models
        sample_counts: List of sample counts for each model for weighted averaging

    Returns:
        PyTorchSGDClassifier: Merged model
    """
    if not models:
        raise ValueError("No models provided for federated averaging")

    # Create a new model with the same architecture as the first model
    reference_model = models[0]

    # Initialize with the same hyperparameters
    global_model = PyTorchSGDClassifier(**reference_model.get_params())

    # Set required attributes
    global_model.classes_ = reference_model.classes_
    global_model.n_features_in_ = reference_model.n_features_in_

    # Initialize the model with the same architecture
    n_classes = len(reference_model.classes_)
    global_model._initialize_model(reference_model.n_features_in_, n_classes)

    # Copy the scaler from the reference model
    global_model.scaler = reference_model.scaler
    global_model.scaler_fitted_ = hasattr(reference_model, 'scaler_fitted_')

    # Calculate the total number of samples
    total_samples = sum(sample_counts)

    # Create dictionary to hold the sums of parameters
    global_dict = {}
    for name, param in global_model.model.state_dict().items():
        global_dict[name] = torch.zeros_like(param)

    # Weighted average of parameters
    for model, sample_count in zip(models, sample_counts):
        weight = sample_count / total_samples
        for name, param in model.model.state_dict().items():
            global_dict[name] += param.data * weight

    # Update the global model with the averaged parameters
    global_model.model.load_state_dict(global_dict)

    return global_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data

    Args:
        model: The model to evaluate
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Evaluation metrics
    """
    import sklearn.metrics as metrics

    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Check if binary or multiclass for ROC AUC
    if len(np.unique(y_test)) == 2:
        # Binary classification
        roc_auc = metrics.roc_auc_score(y_test, y_pred_proba[:, 1])
        precision = metrics.precision_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
    else:
        # Multiclass
        roc_auc = metrics.roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    # Return metrics
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
