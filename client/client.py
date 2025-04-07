import os
import json
import time
import logging
import requests
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from platform_utils import detect_platform, optimize_rf_params

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("client")

# Get environment variables
CLIENT_ID = os.getenv("CLIENT_ID", "default")
DATASET_PATH = os.getenv("DATASET_PATH", "/data")
CENTRAL_SERVER = os.getenv("CENTRAL_SERVER", "http://central:8080")
DATA_SOURCE = os.getenv("DATA_SOURCE", "mnist")  # Options: "mnist", "parquet"
PARQUET_FILE = os.getenv("PARQUET_FILE", "")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")

# Detect platform and get optimized parameters
platform_config = detect_platform()
logger.info(f"Running on detected platform: {platform_config['platform']}")

# Training parameters
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds


def load_data_from_parquet(file_path, target_column):
    """Load data from a Parquet file"""
    logger.info(f"Loading data from Parquet file: {file_path}")

    try:
        # Read the Parquet file
        df = pd.read_parquet(file_path)

        # Verify target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in the Parquet file")

        # Extract features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Handle categorical columns by converting to numerical
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            logger.info(f"Converting {len(categorical_cols)} categorical columns to numerical")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Convert to numpy arrays
        X = X.to_numpy().astype('float32')
        y = y.to_numpy()

        logger.info(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")

        return X, y

    except Exception as e:
        logger.error(f"Error loading Parquet file: {e}")
        raise


def load_data():
    """Load dataset for this client and split it based on client ID."""
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Choose data source based on environment variable
    if DATA_SOURCE.lower() == "parquet" and PARQUET_FILE:
        # Load from Parquet file
        X, y = load_data_from_parquet(PARQUET_FILE, TARGET_COLUMN)
    else:
        # Default to MNIST dataset
        # Check if data already exists
        X_path = os.path.join(DATASET_PATH, "X.npy")
        y_path = os.path.join(DATASET_PATH, "y.npy")

        if os.path.exists(X_path) and os.path.exists(y_path):
            logger.info("Loading MNIST data from cache...")
            X = np.load(X_path)
            y = np.load(y_path)
        else:
            # Load MNIST dataset
            logger.info("Downloading MNIST dataset...")
            mnist = fetch_openml('mnist_784', version=1, cache=True)
            X = mnist.data.astype('float32') / 255.0  # Normalize
            y = mnist.target

            # Save data for future use
            np.save(X_path, X)
            np.save(y_path, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further partition training data based on client ID
    total_samples = len(X_train)

    if CLIENT_ID == "client1":
        start_idx = 0
        end_idx = total_samples // 3
    elif CLIENT_ID == "client2":
        start_idx = total_samples // 3
        end_idx = 2 * (total_samples // 3)
    else:  # client3
        start_idx = 2 * (total_samples // 3)
        end_idx = total_samples

    X_train_subset = X_train[start_idx:end_idx]
    y_train_subset = y_train[start_idx:end_idx]

    logger.info(f"Client {CLIENT_ID} loaded {len(X_train_subset)} training examples")

    return (X_train_subset, y_train_subset), (X_test, y_test)


def create_model():
    """Create a new RandomForest model with platform-optimized parameters"""
    rf_params = optimize_rf_params(platform_config)
    logger.info(f"Creating RandomForest with optimized parameters: {rf_params}")
    return RandomForestClassifier(**rf_params)


def train_model(model, X_train, y_train):
    """Train the RandomForest model on local data"""
    logger.info(f"Training RandomForest with {len(X_train)} samples...")

    # Use optimized backend for current platform
    if platform_config['platform'] == 'apple_silicon':
        backend = 'threading'  # Apple Silicon performs well with threading
    elif platform_config['platform'] == 'cuda':
        backend = 'loky'  # Better for CUDA systems
    else:
        backend = 'threading'  # Default

    start_time = time.time()

    # Train with platform-specific optimizations
    with joblib.parallel_backend(backend, n_jobs=platform_config['n_jobs']):
        model.fit(X_train, y_train)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    logger.info("Evaluating model...")

    # Use optimized backend for predictions
    with joblib.parallel_backend('threading', n_jobs=platform_config['n_jobs']):
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
    return {"accuracy": accuracy}


def get_server_status():
    """Get the current status of the federated learning process"""
    try:
        response = requests.get(f"{CENTRAL_SERVER}/status")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get status: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting server status: {e}")
        return None


def get_platform_info():
    """Get platform information from the server"""
    try:
        response = requests.get(f"{CENTRAL_SERVER}/platform")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get platform info: {response.status_code}, {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Error getting platform info: {e}")
        return {}


def get_global_model():
    """Retrieve the global model from the central server"""
    for retry in range(MAX_RETRIES):
        try:
            response = requests.get(f"{CENTRAL_SERVER}/model")
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get model: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error retrieving global model: {e}")

        # Retry after delay
        logger.info(f"Retrying in {RETRY_DELAY} seconds... (Attempt {retry + 1}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)

    raise Exception("Failed to retrieve global model after maximum retries")


def deserialize_model(serialized_params):
    """Deserialize model from a dictionary and create a new RandomForest classifier"""
    # Get platform-optimized parameters
    rf_params = optimize_rf_params(platform_config)

    # Update with parameters from the serialized model
    rf_params.update({
        'n_estimators': serialized_params['params']['n_estimators'],
        'criterion': serialized_params['params']['criterion'],
        'max_depth': serialized_params['params']['max_depth'],
        'min_samples_split': serialized_params['params']['min_samples_split'],
        'min_samples_leaf': serialized_params['params']['min_samples_leaf'],
        'bootstrap': serialized_params['params']['bootstrap'],
    })

    # Create model with optimized parameters
    model = RandomForestClassifier(**rf_params)

    return model


def upload_model(model, sample_count, metrics):
    """Upload the trained model to the central server"""
    # Save model to a temporary file
    os.makedirs(f"{DATASET_PATH}/models", exist_ok=True)
    model_path = f"{DATASET_PATH}/models/model_{CLIENT_ID}_temp.joblib"

    # Use compression for model file if on Apple Silicon (better I/O performance)
    if platform_config['platform'] == 'apple_silicon':
        joblib.dump(model, model_path, compress=3)
    else:
        joblib.dump(model, model_path)

    # Prepare metadata
    metadata = {
        "client_id": CLIENT_ID,
        "sample_count": sample_count,
        "metrics": metrics,
        "platform": platform_config['platform']
    }

    for retry in range(MAX_RETRIES):
        try:
            # Send model file and metadata to server
            with open(model_path, 'rb') as model_file:
                files = {'model_file': model_file}
                response = requests.post(
                    f"{CENTRAL_SERVER}/upload",
                    files=files,
                    data={'json': json.dumps(metadata)},
                    json=metadata
                )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to upload model: {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error uploading model: {e}")

        # Retry after delay
        logger.info(f"Retrying in {RETRY_DELAY} seconds... (Attempt {retry + 1}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)

    raise Exception("Failed to upload model after maximum retries")


def main():
    """Main function to run the client in the federated learning process"""
    # Load data for this client
    (X_train, y_train), (X_test, y_test) = load_data()

    # Create models directory for saving
    os.makedirs(f