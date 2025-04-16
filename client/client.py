import os
import json
import time
import logging
import requests
import numpy as np
import joblib
import pandas as pd
import sys
import traceback

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
logger.info(f"GPU enabled: {platform_config['use_gpu']}")

# Training parameters
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds


def log_memory_usage():
    """Log current memory usage"""
    try:
        import psutil
        usage = psutil.virtual_memory()
        logger.info(
            f"Memory usage: {usage.percent}% ({usage.used / 1024 / 1024:.1f}MB used out of {usage.total / 1024 / 1024:.1f}MB)")
    except ImportError:
        logger.warning("psutil not available, skipping memory usage logging")


def load_data_from_parquet(file_path, target_column):
    """Load data from a Parquet file"""
    logger.info(f"Loading data from Parquet file: {file_path}")
    log_memory_usage()

    try:
        # Read the Parquet file in batches to save memory
        # First get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        logger.info(f"Parquet file size: {file_size:.2f} MB")

        # Initialize variables
        df = None
        num_rows = 0

        # For very large files, use a more conservative approach
        if file_size > 1000:  # More than 1GB
            logger.info("Large file detected, using memory-efficient loading")

            try:
                # Try using PyArrow for memory efficiency
                import pyarrow.parquet as pq

                # First just read schema and metadata
                parquet_file = pq.ParquetFile(file_path)
                num_rows = parquet_file.metadata.num_rows
                logger.info(f"Total rows in Parquet file: {num_rows}")

                # For really large datasets, sample a subset
                if num_rows > 200000:  # Only keep up to 500K rows to avoid memory issues
                    logger.info(f"Very large dataset detected ({num_rows} rows). Using a sample.")
                    sample_ratio = min(1.0, 200000 / num_rows)
                    logger.info(f"Sampling {sample_ratio:.2%} of data")

                    # Read in smaller chunks
                    batches = []
                    total_sampled = 0

                    # Determine how many row groups we have
                    num_row_groups = parquet_file.metadata.num_row_groups
                    logger.info(f"Found {num_row_groups} row groups in file")

                    # If we have multiple row groups, sample from each
                    if num_row_groups > 1:
                        for i in range(num_row_groups):
                            if np.random.random() > sample_ratio:
                                continue  # Skip some row groups based on sample ratio
                            batch = parquet_file.read_row_group(i)
                            batch_df = batch.to_pandas()

                            # Further sample within the batch if needed
                            if len(batch_df) > 50000:  # Limit each batch size
                                batch_df = batch_df.sample(n=50000)

                            batches.append(batch_df)
                            total_sampled += len(batch_df)

                            if total_sampled >= 500000:
                                break  # Stop once we have enough samples

                        if batches:
                            df = pd.concat(batches, ignore_index=True)
                            logger.info(f"Sampled {len(df)} rows from multiple row groups")
                    else:
                        # If only one row group, read it all and sample
                        df = parquet_file.read().to_pandas()
                        # Sample the dataframe
                        if len(df) > 500000:
                            df = df.sample(n=500000, random_state=42)
                        logger.info(f"Sampled {len(df)} rows from single row group")
                else:
                    # For moderately sized datasets, read the whole file
                    df = parquet_file.read().to_pandas()
                    logger.info(f"Loaded all {len(df)} rows")

            except (ImportError, Exception) as e:
                logger.warning(f"PyArrow read failed: {e}. Falling back to pandas.")
                # Fall back to pandas for reading
                df = pd.read_parquet(file_path)
                num_rows = len(df)

                # Sample if too large
                if num_rows > 500000:
                    df = df.sample(n=500000, random_state=42)
                    logger.info(f"Sampled {len(df)} rows from {num_rows} total rows")
        else:
            # For smaller files, read directly using pandas
            df = pd.read_parquet(file_path)
            num_rows = len(df)
            logger.info(f"Loaded all {num_rows} rows directly")

        # Verify df was loaded
        if df is None or len(df) == 0:
            logger.error("Failed to load data or empty dataframe")
            raise ValueError("No data loaded from Parquet file")

        # Verify target column exists
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found. Available columns: {df.columns.tolist()}")
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

        logger.info(f"Final dataset: {X.shape[0]} samples and {X.shape[1]} features")
        log_memory_usage()

        return X, y

    except Exception as e:
        logger.error(f"Error loading Parquet file: {str(e)}")
        logger.error(traceback.format_exc())
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
            try:
                from sklearn.datasets import fetch_openml
                mnist = fetch_openml('mnist_784', version=1, cache=True)
                X = mnist.data.astype('float32') / 255.0  # Normalize
                y = mnist.target
            except Exception as e:
                logger.error(f"Error downloading MNIST: {e}")
                # Create simple dummy data
                logger.info("Creating dummy data instead")
                X = np.random.rand(1000, 784).astype('float32')
                y = np.random.randint(0, 10, size=1000).astype(str)

            # Save data for future use
            np.save(X_path, X)
            np.save(y_path, y)

    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Client {CLIENT_ID} loaded {len(X_train)} training examples and {len(X_test)} test examples")
    log_memory_usage()

    return (X_train, y_train), (X_test, y_test)


def create_model():
    """Create a model optimized for the current platform"""
    model_config = optimize_model_params(platform_config)

    if model_config['model_type'] == 'xgboost':
        try:
            import xgboost as xgb
            logger.info(f"Creating XGBoost model with GPU acceleration")
            return xgb.XGBClassifier(**model_config['params'])
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**model_config['params'])
    else:
        # Default to RandomForest
        from sklearn.ensemble import RandomForestClassifier
        logger.info(f"Creating RandomForest with optimized parameters")
        return RandomForestClassifier(**model_config['params'])


def train_model(model, X_train, y_train):
    """Train the model on local data"""
    logger.info(f"Training model with {len(X_train)} samples...")
    log_memory_usage()

    start_time = time.time()

    # Train based on model type
    if hasattr(model, 'tree_method') and getattr(model, 'tree_method', '') == 'gpu_hist':
        # XGBoost with GPU
        model.fit(X_train, y_train)
    else:
        # Standard RandomForest or other models
        if hasattr(model, 'n_jobs'):  # Enable parallel training if supported
            with joblib.parallel_backend('threading', n_jobs=platform_config['n_jobs']):
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    log_memory_usage()
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    logger.info("Evaluating model...")

    # Use optimized backend for predictions
    start_time = time.time()

    if hasattr(model, 'tree_method') and getattr(model, 'tree_method', '') == 'gpu_hist':
        # XGBoost with GPU
        y_pred = model.predict(X_test)
    else:
        # Standard model
        with joblib.parallel_backend('threading', n_jobs=platform_config['n_jobs']):
            y_pred = model.predict(X_test)

    eval_time = time.time() - start_time

    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f} (evaluation took {eval_time:.2f} seconds)")
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
    """Deserialize model from a dictionary and create a new model"""
    model_type = serialized_params.get('model_type', 'randomforest')

    if model_type == 'xgboost' and platform_config['use_gpu']:
        try:
            import xgboost as xgb
            # Adjust parameters for current platform
            params = serialized_params['params'].copy()
            if platform_config['use_gpu']:
                params['tree_method'] = 'gpu_hist'
                params['gpu_id'] = 0
            model = xgb.XGBClassifier(**params)
            logger.info("Created XGBoost model from server parameters")
            return model
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForest")

    # Default to RandomForest
    from sklearn.ensemble import RandomForestClassifier
    # Get optimized parameters for this platform but use estimator count from server
    rf_params = optimize_model_params(platform_config)['params']
    if 'n_estimators' in serialized_params['params']:
        rf_params['n_estimators'] = serialized_params['params']['n_estimators']
    model = RandomForestClassifier(**rf_params)
    logger.info("Created RandomForest model from server parameters")
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
        "platform": platform_config['platform'],
        "use_gpu": platform_config['use_gpu']
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
    # Log GPU status
    if platform_config['use_gpu']:
        logger.info(f"GPU acceleration enabled: {platform_config['gpu_count']} GPUs detected")

        # Check GPU devices if PyTorch is available
        try:
            import torch
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        except ImportError:
            pass

    try:
        # Load data for this client
        (X_train, y_train), (X_test, y_test) = load_data()

        # Create models directory for saving
        os.makedirs(f"{DATASET_PATH}/models", exist_ok=True)

        # Wait for server to be ready
        logger.info("Waiting for server to be ready...")
        while True:
            status = get_server_status()
            if status:
                logger.info(f"Server is ready. Current round: {status['current_round']}")
                logger.info(f"Server model type: {status.get('model_type', 'unknown')}")
                logger.info(f"Server GPU enabled: {status.get('gpu_enabled', False)}")
                break
            time.sleep(5)

        # Participate in federated learning rounds
        last_round = -1

        while True:
            # Get current server status
            status = get_server_status()
            if not status:
                logger.warning("Could not get server status. Retrying...")
                time.sleep(5)
                continue

            current_round = status['current_round']

            # Check if training is complete
            if status['status'] == 'completed':
                logger.info("Federated learning process has completed.")
                break

            # Skip if we've already done this round
            if current_round == last_round:
                logger.info(f"Waiting for next round... (Current: {current_round})")
                time.sleep(5)
                continue

            logger.info(f"Starting round {current_round}")

            # Get the global model parameters
            try:
                global_model_params = get_global_model()

                # For first round, create a new model, otherwise start with global model params
                if current_round == 0:
                    model = create_model()
                else:
                    # Create model from server parameters
                    model = deserialize_model(global_model_params)

                logger.info("Initialized model for this round")
            except Exception as e:
                logger.error(f"Error initializing model: {e}")
                time.sleep(5)
                continue

            # Train the model locally
            try:
                model = train_model(model, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training model: {e}")
                logger.error(traceback.format_exc())
                continue

            # Evaluate the model
            try:
                metrics = evaluate_model(model, X_test, y_test)
            except Exception as e:
                logger.error(f"Error evaluating model: {e}")
                metrics = {"accuracy": 0.0}

            # Save model checkpoint
            checkpoint_path = f"{DATASET_PATH}/models/model_{CLIENT_ID}_round_{current_round}.joblib"
            joblib.dump(model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Upload the trained model
            try:
                response = upload_model(model, len(X_train), metrics)
                logger.info(f"Model uploaded successfully: {response}")
                last_round = current_round
            except Exception as e:
                logger.error(f"Failed to upload model: {e}")

            # Small delay before next iteration
            time.sleep(2)

        # Training complete - save final model
        final_model_path = f"{DATASET_PATH}/models/final_model_{CLIENT_ID}.joblib"
        joblib.dump(model, final_model_path)
        logger.info(f"Final model saved to {final_model_path}")

        # Wait a bit before exiting to ensure logs are flushed
        time.sleep(2)
        logger.info("Client process complete")

    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()