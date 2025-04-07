import os
import json
import time
import logging
import threading
import numpy as np
import joblib
from flask import Flask, jsonify, request, Response
from model import create_global_model, serialize_model, merge_forest_weights
from sklearn.ensemble import RandomForestClassifier
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platform_utils import detect_platform, optimize_rf_params

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("central_server")

# Detect platform and get optimized parameters
platform_config = detect_platform()
logger.info(f"Running on detected platform: {platform_config['platform']}")

# Initialize Flask app
app = Flask(__name__)

# Global variables for FL process
global_model = None
client_models = {}
current_round = 0
total_rounds = 10
min_clients = 3
is_training_complete = False
round_metrics = []
lock = threading.Lock()  # Thread lock for model updates


# Initialize the global model with platform-optimized parameters
def initialize_global_model():
    global global_model
    rf_params = optimize_rf_params(platform_config)
    logger.info(f"Initializing global model with parameters: {rf_params}")
    global_model = create_global_model(rf_params)


def deserialize_model(serialized_params):
    """Deserialize model from a dictionary and create a new RandomForest classifier"""
    # Create base model with the same parameters
    rf_params = optimize_rf_params(platform_config)
    rf_params.update({
        'n_estimators': serialized_params['params']['n_estimators'],
        'criterion': serialized_params['params']['criterion'],
        'max_depth': serialized_params['params']['max_depth'],
        'min_samples_split': serialized_params['params']['min_samples_split'],
        'min_samples_leaf': serialized_params['params']['min_samples_leaf'],
        'bootstrap': serialized_params['params']['bootstrap'],
    })

    model = RandomForestClassifier(**rf_params)

    # Manually set classes
    model.classes_ = np.array(serialized_params['classes'])
    model.n_classes_ = serialized_params['n_classes']
    model.n_features_in_ = serialized_params['n_features']

    # Fit a simple dataset to initialize internal structures
    model.fit([[0] * model.n_features_in_], [model.classes_[0]])

    return model


def federated_averaging():
    """Perform federated averaging on collected client models"""
    global global_model, client_models, current_round, is_training_complete

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
                else:
                    logger.warning(f"Model file for {client_id} not found")
            except Exception as e:
                logger.error(f"Error loading model for {client_id}: {e}")

        if not models:
            logger.warning("No valid models to aggregate")
            return

        # Merge models using weighted voting
        try:
            # Apply platform-specific optimizations to model merging
            with joblib.parallel_backend('threading', n_jobs=platform_config['n_jobs']):
                global_model = merge_forest_weights(models, sample_counts)

            # Save the global model
            os.makedirs("models", exist_ok=True)
            global_model_path = f"models/global_round_{current_round}.joblib"
            joblib.dump(global_model, global_model_path)
            logger.info(f"Saved global model to {global_model_path}")

        except Exception as e:
            logger.error(f"Error merging models: {e}")
            return

        # Save metrics for this round
        current_metrics = {
            "round": current_round,
            "clients": list(client_models.keys()),
            "avg_accuracy": np.mean([m['metrics']['accuracy'] for m in client_models.values() if 'metrics' in m]),
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
        "platform": platform_config['platform']
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

    # For RandomForest, we'll save the model file separately
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