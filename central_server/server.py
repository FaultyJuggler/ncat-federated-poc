import os
import time
import logging
import threading
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import numpy as np
import pickle  # For in-memory model serialization
import base64
from flask import Flask, jsonify, request, Response
import sys

from model import PyTorchSGDClassifier

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
model_type = 'pytorch_sgd'  # Default model type
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


# Initialize the global model
def initialize_global_model(X_sample=None, y_sample=None):
    global global_model, model_type

    # If no sample data is provided, create more representative placeholder data
    if X_sample is None or y_sample is None:
        # Create a larger sample that better represents the scale of your data
        n_features = 9  # Adjust based on your actual feature count
        n_samples = 1000  # Much larger sample size for better initialization

        # Create synthetic features with reasonable distributions
        X_sample = np.random.randn(n_samples, n_features)

        # Create labels with similar class imbalance to your real data (75% class 0, 25% class 1)
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
            'early_stopping': False,  # false for partial_fit
            'validation_fraction': 0.1,  # Use 10% of data for early stopping
            'n_iter_no_change': 5,  # Number of iterations with no improvement
            'learning_rate': 'optimal'  # Automatically adjusts based on regularization
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

        # Create the PyTorchSGDClassifier without initialization
        global_model = PyTorchSGDClassifier(
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=100,
            tol=1e-3,
            random_state=42,
            learning_rate=0.01,
            batch_size=128
        )

        # Manually patch the _initialize_model method before calling it
        def patched_initialize_model(self, n_features, n_classes):
            """Fixed version of _initialize_model that avoids the 'out_features' name error"""
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

        # Replace the original method with our patched version
        import types
        global_model._initialize_model = types.MethodType(patched_initialize_model, global_model)

        # Set required attributes
        global_model.n_features_in_ = n_features
        global_model.classes_ = unique_classes

        # Initialize the model
        global_model._initialize_model(n_features, n_classes)
        logger.info("PyTorch model initialization successful")

        # Fit the model with a few epochs on the sample data
        for epoch in range(3):
            global_model.partial_fit(X_sample, y_sample, classes=unique_classes)

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
                    global_model = RandomForestClassifier(**rf_params)
                    global_model.fit(X_sample, y_sample)  # RandomForest doesn't use partial_fit
                    model_type = 'randomforest'
                    logger.info("RandomForest model initialization successful")
                except Exception as rf_error:
                    logger.error(f"RandomForest initialization failed: {rf_error}")
                    global_model = create_sgd_model(model_config)
                    global_model.partial_fit(X_sample, y_sample, classes=unique_classes)
                    model_type = 'sgd'
                    logger.info("SGD model initialization successful as fallback")
            else:
                # Default to SGD
                global_model = create_sgd_model(model_config)
                global_model.partial_fit(X_sample, y_sample, classes=unique_classes)
                model_type = 'sgd'
                logger.info("SGD model initialization successful")

        except Exception as sgd_error:
            logger.error(f"SGD model initialization also failed: {sgd_error}")
            raise RuntimeError("Failed to initialize any model type")

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


# def merge_models(models, sample_counts):
#     """
#     Merge models using weighted voting based on sample counts.
#     Handles different model types appropriately.
#     """
#     # Basic validation
#     if not models:
#         raise ValueError("No models provided for merging")
#
#     if not all(type(models[0]) == type(m) for m in models):
#         logger.warning("Inconsistent model types detected. Converting to same type.")
#
#     # Get model type from first model
#     model_type = 'sgd'  # Default to SGD
#
#     # Check if the model is SGDClassifier (recommended approach)
#     if hasattr(models[0], '_finalize_coef'):  # SGDClassifier has this internal method
#         logger.info("Merging SGD Classifier models")
#         model_type = 'sgd'
#     # Fallback checks for other model types
#     elif hasattr(models[0], 'estimators_'):
#         logger.warning("RandomForest model detected, but SGD is recommended")
#         model_type = 'randomforest'
#
#     # Create a new global model with optimized parameters for current platform
#     model_config = optimize_model_params(platform_config)
#
#     # Override model_type to ensure SGD is used
#     model_config['model_type'] = 'sgd'
#
#     if model_type == 'sgd':
#         # For SGDClassifier
#         from sklearn.linear_model import SGDClassifier
#         global_model = SGDClassifier(**model_config['params'])
#
#         # Merge SGD models by averaging coefficients and intercepts
#         if models and all(hasattr(m, 'coef_') for m in models):
#             # Calculate weights based on sample counts
#             total_samples = sum(sample_counts)
#             weights = [count / total_samples for count in sample_counts]
#
#             # Get first model's shapes to initialize
#             reference_model = models[0]
#             merged_coef = np.zeros_like(reference_model.coef_)
#             merged_intercept = np.zeros_like(reference_model.intercept_)
#
#             # Weighted average of coefficients and intercepts
#             for model, weight in zip(models, weights):
#                 merged_coef += model.coef_ * weight
#                 merged_intercept += model.intercept_ * weight
#
#             # Set the averaged parameters to the global model
#             global_model.coef_ = merged_coef
#             global_model.intercept_ = merged_intercept
#
#             # Copy other necessary attributes from reference model
#             global_model.classes_ = reference_model.classes_
#             global_model.n_features_in_ = reference_model.n_features_in_
#
#             # Set the model as fitted
#             global_model._fitted = True
#         else:
#             logger.warning("Some SGD models don't have coefficients. Using model with most data.")
#             global_model = models[np.argmax(sample_counts)]
#
#     else:
#         # For RandomForest (fallback case)
#         logger.warning("Using RandomForest model instead of recommended SGD Classifier")
#         from sklearn.ensemble import RandomForestClassifier
#         global_model = RandomForestClassifier(**model_config['params'])
#
#         # Calculate the total number of trees and weights per client
#         total_samples = sum(sample_counts)
#         weights = [count / total_samples for count in sample_counts]
#
#         # Calculate how many trees to take from each client model
#         total_trees = global_model.n_estimators
#         trees_per_model = [int(round(weight * total_trees)) for weight in weights]
#
#         # Adjust to ensure we get exactly total_trees
#         while sum(trees_per_model) < total_trees:
#             idx = trees_per_model.index(min(trees_per_model))
#             trees_per_model[idx] += 1
#         while sum(trees_per_model) > total_trees:
#             idx = trees_per_model.index(max(trees_per_model))
#             trees_per_model[idx] -= 1
#
#         # Gather trees from each model according to their weights
#         merged_estimators = []
#         for model, n_trees in zip(models, trees_per_model):
#             if n_trees <= 0:
#                 continue
#             # For simplicity, take the first n_trees
#             merged_estimators.extend(model.estimators_[:n_trees])
#
#         # Update the global model's estimators
#         global_model.estimators_ = merged_estimators
#
#         # Ensure other attributes are correctly set
#         if models:
#             reference_model = models[0]
#             global_model.n_classes_ = reference_model.n_classes_
#             global_model.classes_ = reference_model.classes_
#             global_model.n_features_in_ = reference_model.n_features_in_
#
#     return global_model
#
#
# def federated_averaging():
#     global global_model, current_round, client_models
#
#     logger.info(f"Performing federated averaging for round {current_round}")
#
#     if len(client_models) == 0:
#         logger.warning("No client models available for aggregation")
#         return False
#
#     # Collect models and their weights (sample counts)
#     models = []
#     weights = []
#
#     # Use the in-memory models directly from client_models dictionary
#     for client_id, client_data in client_models.items():
#         if 'model' in client_data and client_data['model'] is not None:
#             models.append(client_data['model'])
#             weights.append(client_data['sample_count'])
#             logger.info(f"Using in-memory model from client {client_id} with weight {client_data['sample_count']}")
#         else:
#             logger.warning(f"No valid model found for client {client_id}")
#
#     if len(models) == 0:
#         logger.warning("No valid models to aggregate")
#         return False
#
#     try:
#         # Normalize weights to sum to 1.0
#         total_samples = sum(weights)
#         weights = [w / total_samples for w in weights]
#
#         # Check if model has coefficient and intercept attributes (like sklearn models)
#         if hasattr(models[0], 'coef_') and hasattr(models[0], 'intercept_'):
#             # Create a new model with the same parameters
#             aggregated_model = clone(models[0])
#
#             # Average the coefficients
#             coefs = np.array([model.coef_ for model in models])
#             aggregated_model.coef_ = np.sum([coef * weight for coef, weight in zip(coefs, weights)], axis=0)
#
#             # Average the intercepts
#             intercepts = np.array([model.intercept_ for model in models])
#             aggregated_model.intercept_ = np.sum([intercept * weight for intercept, weight in zip(intercepts, weights)],
#                                                  axis=0)
#
#             global_model = aggregated_model
#             logger.info("Successfully created aggregated model by averaging coefficients and intercepts")
#         else:
#             # For custom model types, use your existing merge_models function
#             logger.info("Models don't have standard coef_/intercept_ attributes, using custom merge function")
#             global_model = merge_models(models, weights)
#
#         # Make sure the global model is valid
#         if global_model is not None:
#             # Define data directory and ensure it exists
#             data_dir = os.path.join(os.path.dirname(__file__), "data")
#             os.makedirs(data_dir, exist_ok=True)
#
#             # Save the model
#             model_path = os.path.join(data_dir, f"global_model_round_{current_round}.joblib")
#             joblib.dump(global_model, model_path)
#             logger.info(f"Saved aggregated global model for round {current_round} to {model_path}")
#             return True
#         else:
#             logger.error("Failed to create a valid global model")
#             return False
#
#     except Exception as e:
#         logger.error(f"Error during federated averaging: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return False



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