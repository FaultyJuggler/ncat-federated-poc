import os
import json
import time
import logging
import threading
import numpy as np
import joblib
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
model_type = 'randomforest'  # Default model type
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
def initialize_global_model():
    global global_model, model_type

    # Force SGDClassifier regardless of what platform_utils recommends
    from sklearn.linear_model import SGDClassifier
    global_model = SGDClassifier(
        loss='log_loss',
        alpha=0.0001,
        max_iter=5,
        tol=0.001,
        random_state=42,
        warm_start=True
    )
    model_type = 'sgd'
    logger.info("Initialized SGDClassifier model for memory efficiency")


def serialize_model(model):
    """Serialize model to a dictionary"""
    serialized = {}

    # Store model type
    if hasattr(model, 'tree_method') and getattr(model, 'tree_method', '') == 'gpu_hist':
        serialized['model_type'] = 'xgboost'
    else:
        serialized['model_type'] = 'sgd'  # or 'randomforest' depending on your model

    # For XGBoost models
    if serialized['model_type'] == 'xgboost':
        try:
            import xgboost as xgb
            if isinstance(model, xgb.XGBClassifier):
                # XGBoost models have different attributes
                serialized['params'] = model.get_params()

                # XGBoost might not have n_classes_ attribute yet (before fit)
                if hasattr(model, 'n_classes_'):
                    serialized['n_classes'] = model.n_classes_
                else:
                    # Default to binary classification if not known
                    serialized['n_classes'] = 2

                if hasattr(model, 'n_features_in_'):
                    serialized['n_features'] = model.n_features_in_
                else:
                    serialized['n_features'] = 0

                if hasattr(model, 'classes_'):
                    serialized['classes'] = model.classes_.tolist()
                else:
                    serialized['classes'] = None
        except Exception as e:
            logger.error(f"Error serializing XGBoost model: {str(e)}")
            # Provide defaults
            serialized['params'] = {}
            serialized['n_classes'] = 2
            serialized['n_features'] = 0
            serialized['classes'] = None
    elif serialized['model_type'] == 'sgd':
        # For SGD Classifier
        serialized['n_classes'] = model.classes_.shape[0] if hasattr(model, 'classes_') else 2
        serialized['n_features'] = model.n_features_in_ if hasattr(model, 'n_features_in_') else 0
        serialized['classes'] = model.classes_.tolist() if hasattr(model, 'classes_') else None
        serialized['params'] = {
            'loss': getattr(model, 'loss', 'log_loss'),
            'penalty': getattr(model, 'penalty', 'l2'),
            'alpha': getattr(model, 'alpha', 0.0001),
            'max_iter': getattr(model, 'max_iter', 5),
            'tol': getattr(model, 'tol', 0.001),
            'random_state': getattr(model, 'random_state', 42),
            'warm_start': getattr(model, 'warm_start', True)
        }
    else:
        # RandomForest serialization (original code)
        serialized['n_classes'] = model.n_classes_ if hasattr(model, 'n_classes_') else 2
        serialized['n_features'] = model.n_features_in_ if hasattr(model, 'n_features_in_') else 0
        serialized['classes'] = model.classes_.tolist() if hasattr(model, 'classes_') else None
        serialized['params'] = {
            'n_estimators': getattr(model, 'n_estimators', 100),
            'criterion': getattr(model, 'criterion', 'gini'),
            'max_depth': getattr(model, 'max_depth', None),
            'min_samples_split': getattr(model, 'min_samples_split', 2),
            'min_samples_leaf': getattr(model, 'min_samples_leaf', 1),
            'bootstrap': getattr(model, 'bootstrap', True)
        }

    return serialized


def deserialize_model(serialized_params):
    """Deserialize model from a dictionary and create a new model"""
    model_type = serialized_params.get('model_type', 'randomforest')

    if model_type == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(**serialized_params['params'])
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(**serialized_params['params'])
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
    if hasattr(models[0], 'tree_method') and getattr(models[0], 'tree_method', '') == 'gpu_hist':
        logger.info("Merging XGBoost models")
        model_type = 'xgboost'
    else:
        logger.info("Merging RandomForest models")
        model_type = 'randomforest'

    # Create a new global model with optimized parameters for current platform
    model_config = optimize_model_params(platform_config)

    if model_type == 'xgboost':
        try:
            import xgboost as xgb
            global_model = xgb.XGBClassifier(**model_config['params'])

            # For XGBoost, we need to train a new model on weighted predictions
            # This is a simplified approach - a production system would be more sophisticated
            # Here we're just doing a weighted average of the models
            return models[np.argmax(sample_counts)]  # Return the model from client with most data

        except ImportError:
            logger.warning("XGBoost not available for merging. Falling back to RandomForest.")
            from sklearn.ensemble import RandomForestClassifier
            model_config = {'model_type': 'randomforest', 'params': optimize_rf_params(platform_config)}
            global_model = RandomForestClassifier(**model_config['params'])
    else:
        # For RandomForest, we'll merge trees as before
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
    """Perform federated model merging on collected client models"""
    global global_model, client_models, current_round, is_training_complete, model_type

    with lock:
        if len(client_models) < min_clients:
            logger.info(f"Waiting for more clients. Current: {len(client_models)}")
            return

        logger.info(f"Performing federated averaging for round {current_round}")

        # Extract models and sample counts
        models = []
        sample_counts = []

        for client_id, model_info in client_models.items():
            try:
                # Reload model from the saved file
                model_path = f"models/{client_id}_round_{current_round}.joblib"
                if os.path.exists(model_path):
                    client_model = joblib.load(model_path)
                    models.append(client_model)
                    sample_counts.append(model_info['sample_count'])

                    # Update model type based on received model
                    if hasattr(client_model, 'tree_method') and getattr(client_model, 'tree_method', '') == 'gpu_hist':
                        model_type = 'xgboost'
                else:
                    logger.warning(f"Model file for {client_id} not found")
            except Exception as e:
                logger.error(f"Error loading model for {client_id}: {e}")

        if not models:
            logger.warning("No valid models to aggregate")
            return

        # Merge models using weighted voting
        try:
            global_model = merge_models(models, sample_counts)

            # Save the global model
            os.makedirs("models", exist_ok=True)
            global_model_path = f"models/global_round_{current_round}.joblib"
            joblib.dump(global_model, global_model_path)
            logger.info(f"Saved global model to {global_model_path}")

        except Exception as e:
            logger.error(f"Error merging models: {e}")
            logger.error(f"Exception details: {str(e)}")
            traceback.print_exc()
            return

        # Save metrics for this round
        current_metrics = {
            "round": current_round,
            "clients": list(client_models.keys()),
            "avg_accuracy": np.mean([m['metrics']['accuracy'] for m in client_models.values() if 'metrics' in m]),
            "model_type": model_type
        }
        round_metrics.append(current_metrics)
        logger.info(f"Round {current_round} metrics: {current_metrics}")

        # Reset client models for next round
        client_models.clear()

        # Increment round counter
        current_round += 1

        # Check if training is complete
        if current_round >= total_rounds:
            logger.info(f"Federated learning completed after {total_rounds} rounds")
            is_training_complete = True
            # Save the final model
            joblib.dump(global_model, "final_model.joblib")


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
    """Receive a trained model from a client"""
    # Extract metadata
    data = request.json
    client_id = data['client_id']
    sample_count = data['sample_count']
    metrics = data.get('metrics', {})

    logger.info(f"Received model metadata from client {client_id} with {sample_count} samples")

    # For model types like XGBoost and RandomForest, we'll save the model file separately
    if 'model_file' in request.files:
        model_file = request.files['model_file']
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{client_id}_round_{current_round}.joblib"
        model_file.save(model_path)
        logger.info(f"Saved model file to {model_path}")

    # Store the client model metadata
    with lock:
        client_models[client_id] = {
            'sample_count': sample_count,
            'metrics': metrics,
            'timestamp': time.time()
        }

    # Check if we should perform federated averaging
    if len(client_models) >= min_clients:
        # Start a new thread to perform federated averaging
        threading.Thread(target=federated_averaging).start()

    return jsonify({
        "status": "success",
        "current_round": current_round,
        "message": f"Model received from client {client_id}"
    })


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