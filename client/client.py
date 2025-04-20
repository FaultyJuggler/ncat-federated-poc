import numpy as np
import pandas as pd
import os
import sys
import json
import logging
import pickle
import requests
import time
import gc
import platform
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib  # Make sure this import is here

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('client')

# Configuration
CLIENT_ID = os.environ.get('CLIENT_ID', 'client1')
DATASET_PATH = os.environ.get('DATASET_PATH', '/app/data')
CENTRAL_SERVER = os.environ.get('CENTRAL_SERVER', 'http://central:5000')
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'parquet')
PARQUET_FILE = os.environ.get('PARQUET_FILE', 'adult.parquet')
TARGET_COLUMN = os.environ.get('TARGET_COLUMN', 'income')
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '10000'))
USE_PARTIAL_FIT = os.environ.get('USE_PARTIAL_FIT', 'True').lower() == 'true'

# Platform-specific config
platform_config = {
    'use_gpu': os.environ.get('USE_GPU', 'False').lower() == 'true',
}

# Network retry settings
MAX_RETRIES = 5
RETRY_DELAY = 2

# Create a PyTorch-based SGD classifier wrapper - Moved to top level
try:
    import torch


    class PyTorchSGDClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, loss='log', penalty='l2', alpha=0.0001,
                     max_iter=1000, tol=1e-3, random_state=42):
            self.loss = loss
            self.penalty = penalty
            self.alpha = alpha
            self.max_iter = max_iter
            self.tol = tol
            self.random_state = random_state
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = None
            self.classes_ = None

        def fit(self, X, y):
            # Implementation would go here
            pass

        def predict(self, X):
            # Implementation would go here
            pass

        def partial_fit(self, X, y, classes=None):
            # Implementation would go here
            pass
except ImportError:
    # Define empty class if torch is not available to prevent errors
    class PyTorchSGDClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is not available")


def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())

    # Get memory info
    mem_info = process.memory_info()

    # Calculate usage in MB
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)

    # Get system memory info
    system_mem = psutil.virtual_memory()
    system_total_mb = system_mem.total / (1024 * 1024)
    system_available_mb = system_mem.available / (1024 * 1024)

    logger.info(f"Memory Usage - Process: {rss_mb:.2f} MB (RSS), {vms_mb:.2f} MB (VMS), "
                f"System: {system_available_mb:.2f} MB available of {system_total_mb:.2f} MB total")


class BatchProcessor:
    """
    Handles efficient batch processing of datasets for ML training.
    """

    def __init__(self, file_path, target_column, batch_size=10000):
        """
        Initialize the BatchProcessor.

        Args:
            file_path: Path to the parquet file
            target_column: Name of the target column
            batch_size: Size of batches to process at once
        """
        self.file_path = file_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.total_rows = None
        self.feature_cols = None
        self.categorical_cols = None
        self.num_features = None
        self.has_initialized = False

    def initialize(self):
        """
        Perform initialization tasks:
        - Analyze parquet file metadata
        - Determine features and data types
        - Set up preprocessing if needed
        """
        if self.has_initialized:
            return

        logger.info(f"Initializing BatchProcessor for {self.file_path}")

        try:
            # Open the parquet file to get metadata
            parquet_file = pq.ParquetFile(self.file_path)

            # Get the schema
            schema = parquet_file.schema

            # Extract column names and types
            all_columns = [field.name for field in schema]

            # Exclude the target column from features
            if self.target_column in all_columns:
                self.feature_cols = [col for col in all_columns if col != self.target_column]
            else:
                logger.error(f"Target column '{self.target_column}' not found in dataset")
                raise ValueError(f"Target column '{self.target_column}' not found in dataset")

            # Determine categorical columns by checking data types
            # For this simple version, we assume string columns are categorical
            self.categorical_cols = []
            for col in self.feature_cols:
                field = schema.field(col)
                if pa.types.is_string(field.type):
                    self.categorical_cols.append(col)

            # Count total rows if possible from metadata
            metadata = parquet_file.metadata
            if metadata:
                self.total_rows = metadata.num_rows

            # If rows not in metadata, need to calculate it
            if not self.total_rows:
                logger.info("Row count not in metadata, counting rows...")
                self.total_rows = sum(1 for _ in self._read_parquet_chunks())

            # Determine the number of features after one-hot encoding
            # Need to read the first batch to determine this
            for batch_df in self._read_parquet_chunks():
                # Only process the first batch
                X, _ = self._prepare_batch(batch_df)
                self.num_features = X.shape[1]
                break

            logger.info(f"Initialization complete: {len(self.feature_cols)} columns, "
                        f"{len(self.categorical_cols)} categorical, "
                        f"{self.num_features} features after preprocessing, "
                        f"{self.total_rows} total rows")

            self.has_initialized = True

        except Exception as e:
            logger.error(f"Error initializing BatchProcessor: {e}")
            raise

    def get_total_rows(self):
        """
        Return the total number of rows in the dataset.
        """
        if not self.has_initialized:
            self.initialize()

        return self.total_rows

    def get_num_features(self):
        """
        Return the number of features after preprocessing.
        """
        if not self.has_initialized:
            self.initialize()

        return self.num_features

    def _read_parquet_chunks(self):
        """
        Generator that reads the parquet file in chunks.

        Yields:
            DataFrame: pandas DataFrame containing a batch of data
        """
        try:
            # Memory-efficient way to read parquet file in chunks
            parquet_file = pq.ParquetFile(self.file_path)

            # Get the schema
            schema = parquet_file.schema

            # Determine the number of row groups
            num_row_groups = parquet_file.num_row_groups

            # Process each row group
            for i in range(num_row_groups):
                # Read a single row group
                table = parquet_file.read_row_group(i)

                # Convert to pandas DataFrame
                df = table.to_pandas()

                # Ensure the target column exists
                if self.target_column not in df.columns:
                    logger.error(f"Target column '{self.target_column}' not found in row group {i}")
                    continue

                # Process in batch_size chunks to maintain consistent batch sizes
                for start_idx in range(0, len(df), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(df))
                    yield df.iloc[start_idx:end_idx].copy()

        except Exception as e:
            logger.error(f"Error reading parquet file: {e}")
            raise

    def _prepare_batch(self, batch_df):
        """
        Prepare a batch for ML processing:
        - Split features and target
        - Handle categorical features (one-hot encoding)
        - Convert to numpy arrays

        Args:
            batch_df: Pandas DataFrame containing a batch of data

        Returns:
            tuple: (X, y) where X is feature matrix and y is target vector
        """
        # Separate features and target
        X_df = batch_df[self.feature_cols]
        y = batch_df[self.target_column].values

        # One-hot encode categorical features
        if self.categorical_cols:
            X_df = pd.get_dummies(X_df, columns=self.categorical_cols, drop_first=True)

        # Convert to numpy array
        X = X_df.values

        return X, y

    def batch_iterator(self):
        """
        Generate batches of preprocessed data.

        Yields:
            tuple: (X, y, batch_info) where:
                - X is feature matrix
                - y is target vector
                - batch_info is a dict with metadata about the batch
        """
        if not self.has_initialized:
            self.initialize()

        logger.info(f"Starting batch iteration with batch size {self.batch_size}")

        batch_count = 0
        total_rows_processed = 0

        try:
            for batch_df in self._read_parquet_chunks():
                # Prepare the batch
                X, y = self._prepare_batch(batch_df)

                # Create batch info
                batch_info = {
                    'batch_index': batch_count,
                    'batch_size': len(y),
                    'batch_columns': list(batch_df.columns),
                    'batch_categorical': self.categorical_cols.copy() if self.categorical_cols else [],
                }

                # Update counters
                batch_count += 1
                total_rows_processed += len(y)

                # Log progress
                if batch_count % 10 == 0:
                    progress = (total_rows_processed / self.total_rows) * 100
                    logger.info(f"Processed {batch_count} batches, {total_rows_processed} rows "
                                f"({progress:.1f}% complete)")
                    log_memory_usage()

                yield X, y, batch_info

            logger.info(f"Batch iteration complete: {batch_count} batches, {total_rows_processed} rows")

        except Exception as e:
            logger.error(f"Error during batch iteration: {e}")
            raise


def batch_iterator_memory_safe(file_path, target_column, batch_size):
    """
    A memory-safe iterator that processes data in batches.

    This function is a wrapper around BatchProcessor for simpler use cases.

    Args:
        file_path: Path to the parquet file
        target_column: Name of the target column
        batch_size: Size of batches to process

    Yields:
        tuple: (X, y) pairs for each batch
    """
    processor = BatchProcessor(file_path, target_column, batch_size)

    for X, y, _ in processor.batch_iterator():
        yield X, y


def get_unique_classes(file_path, target_column, batch_size=10000):
    """
    Extract unique classes from the dataset.

    Args:
        file_path: Path to the parquet file
        target_column: Name of the target column
        batch_size: Size of batches to process

    Returns:
        array: Array of unique class values
    """
    logger.info(f"Extracting unique classes from {file_path}")
    unique_classes = set()

    # Process in batches to handle large datasets
    for _, y in batch_iterator_memory_safe(file_path, target_column, batch_size):
        # Update set with new unique values
        unique_classes.update(np.unique(y))

    logger.info(f"Found {len(unique_classes)} unique classes: {unique_classes}")
    return np.array(sorted(list(unique_classes)))


def train_model_in_batches(model, file_path, target_column, batch_size=10000):
    """
    Train a model using batches to handle large datasets.

    Args:
        model: The model to train (must support fit or partial_fit)
        file_path: Path to the parquet file
        target_column: Name of the target column
        batch_size: Size of batches to train on

    Returns:
        model: The trained model
        dict: Training metrics
    """
    logger.info(f"Starting batch training with batch size {batch_size}")
    start_time = time.time()

    # Counters for tracking progress
    total_rows = 0
    batch_count = 0

    # If USE_PARTIAL_FIT is True, we'll use partial_fit for incremental training
    # First, we need to get unique classes for partial_fit
    if USE_PARTIAL_FIT and hasattr(model, 'partial_fit'):
        classes = get_unique_classes(file_path, target_column, batch_size)

        # Train on each batch incrementally
        for X, y in batch_iterator_memory_safe(file_path, target_column, batch_size):
            if batch_count == 0:
                # First batch call includes classes
                model.partial_fit(X, y, classes=classes)
            else:
                model.partial_fit(X, y)

            total_rows += len(y)
            batch_count += 1

            # Log progress
            if batch_count % 5 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Processed {batch_count} batches, {total_rows} rows in {elapsed:.2f} seconds")
                log_memory_usage()

    else:
        # If partial_fit is not available or not requested, collect all data and use fit
        # Warning: This could use a lot of memory for large datasets
        logger.warning("Using standard fit() method, which may require more memory")

        all_X = []
        all_y = []

        for X, y in batch_iterator_memory_safe(file_path, target_column, batch_size):
            all_X.append(X)
            all_y.append(y)
            total_rows += len(y)
            batch_count += 1

            # Log progress
            if batch_count % 5 == 0:
                logger.info(f"Loaded {batch_count} batches, {total_rows} rows")
                log_memory_usage()

        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        logger.info(f"Training on combined dataset with {X_combined.shape[0]} rows")
        model.fit(X_combined, y_combined)

    # Training complete, calculate metrics
    elapsed = time.time() - start_time
    metrics = {
        'total_time': elapsed,
        'total_rows': total_rows,
        'batches': batch_count,
        'rows_per_second': total_rows / elapsed if elapsed > 0 else 0,
    }

    logger.info(f"Training complete. Processed {total_rows} rows in {elapsed:.2f} seconds")
    return model, metrics


def evaluate_model_in_batches(model, file_path, target_column, batch_size=10000):
    """
    Evaluate model accuracy using batches to handle large datasets.

    Args:
        model: The trained model (must have predict method)
        file_path: Path to the parquet file
        target_column: Name of the target column
        batch_size: Size of batches for evaluation

    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Starting batch evaluation with batch size {batch_size}")
    start_time = time.time()

    # Counters for metrics
    total_rows = 0
    total_correct = 0
    batch_count = 0

    try:
        # Evaluate on each batch
        for X, y in batch_iterator_memory_safe(file_path, target_column, batch_size):
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(X)

                # Calculate batch accuracy
                correct = (predictions == y).sum()
                total_correct += correct
                total_rows += len(y)
                batch_count += 1

                # Log progress
                if batch_count % 5 == 0:
                    batch_accuracy = correct / len(y)
                    logger.info(f"Batch {batch_count} accuracy: {batch_accuracy:.4f}")
            else:
                logger.error("Model does not have predict method")
                raise AttributeError("Model does not have predict method")
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        # Return partial results if possible
        if total_rows > 0:
            accuracy = total_correct / total_rows
            elapsed = time.time() - start_time
            return {
                'accuracy': accuracy,
                'total_rows': total_rows,
                'total_correct': total_correct,
                'evaluation_time': elapsed,
                'error': str(e)
            }
        else:
            raise

    # Calculate overall accuracy
    accuracy = total_correct / total_rows if total_rows > 0 else 0
    elapsed = time.time() - start_time

    metrics = {
        'accuracy': accuracy,
        'total_rows': total_rows,
        'total_correct': total_correct,
        'evaluation_time': elapsed,
        'batches': batch_count
    }

    logger.info(f"Evaluation complete. Accuracy: {accuracy:.4f} on {total_rows} rows")
    return metrics


def load_data_for_batch_processing(dataset_path, target_column, batch_size):
    """
    Prepare the data path and ensure the file exists.

    Args:
        dataset_path: Base path to the dataset directory
        target_column: Name of the target column
        batch_size: Size of batches for processing

    Returns:
        tuple: (file_path, processor) where processor is a BatchProcessor instance
    """
    # Handle different data sources
    if DATA_SOURCE == 'parquet':
        file_path = os.path.join(dataset_path, PARQUET_FILE)

        # Verify the file exists
        if not os.path.exists(file_path):
            logger.error(f"Parquet file not found: {file_path}")
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        logger.info(f"Using parquet file: {file_path}")

        # Create batch processor
        processor = BatchProcessor(file_path, target_column, batch_size)
        processor.initialize()

        # Log dataset info
        logger.info(f"Dataset loaded: {processor.get_total_rows()} rows, {processor.get_num_features()} features")

        return file_path, processor

    elif DATA_SOURCE == 'csv':
        # For future implementation
        logger.error("CSV data source not yet implemented")
        raise NotImplementedError("CSV data source not yet implemented")

    else:
        logger.error(f"Unknown data source: {DATA_SOURCE}")
        raise ValueError(f"Unknown data source: {DATA_SOURCE}")


def create_model():
    """
    Create a model suitable for the federated learning task.

    The model needs to be serializable for federated learning.

    Returns:
        model: A machine learning model
    """
    # For simplicity, we're using SGD Classifier which is efficient for large datasets
    # and supports partial_fit for incremental training
    from sklearn.linear_model import SGDClassifier

    # Check if GPU acceleration is configured
    if platform_config.get('use_gpu', False):
        try:
            # You would implement GPU version here
            logger.info("GPU acceleration requested, but standard CPU model being used")
        except Exception as e:
            logger.warning(f"Error setting up GPU-accelerated model: {e}")
            logger.info("Falling back to CPU-based model")

    # Create SGD Classifier
    model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=100, random_state=42)
    logger.info("Created SGD Classifier model")

    return model


def get_server_status():
    """
    Check the status of the central server and get current round information.

    Returns:
        dict: Status information from the server
    """
    url = f"{CENTRAL_SERVER}/status"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            status = response.json()
            logger.info(f"Server status: round {status.get('current_round', -1)}, "
                        f"clients: {status.get('clients_submitted', 0)}/{status.get('total_clients', 0)}")
            return status
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error connecting to server (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries exceeded. Unable to connect to server.")
                raise


def get_platform_info():
    """
    Collect information about the running platform for diagnostics.

    Returns:
        dict: Platform information
    """
    # Get basic platform info
    platform_info = {
        'python_version': platform.python_version(),
        'system': platform.system(),
        'release': platform.release(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }

    # Add memory information
    mem = psutil.virtual_memory()
    platform_info['memory_total'] = mem.total / (1024 * 1024)  # MB
    platform_info['memory_available'] = mem.available / (1024 * 1024)  # MB

    # Add CPU information
    platform_info['cpu_cores'] = psutil.cpu_count(logical=False)
    platform_info['cpu_threads'] = psutil.cpu_count(logical=True)

    # Check if GPU is available
    platform_info['gpu_available'] = False
    try:
        import torch
        platform_info['gpu_available'] = torch.cuda.is_available()
        if platform_info['gpu_available']:
            platform_info['gpu_name'] = torch.cuda.get_device_name(0)
            platform_info['gpu_count'] = torch.cuda.device_count()
    except ImportError:
        logger.info("PyTorch not available, cannot check GPU")

    return platform_info


def get_global_model():
    """
    Get the latest global model from the central server.

    Returns:
        tuple: (model, round_number, success_flag)
    """
    url = f"{CENTRAL_SERVER}/model"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Extract model data and round
            data = response.json()
            round_number = data.get('round', -1)

            if 'model' in data:
                # Deserialize the model parameters
                model = deserialize_model(data['model'])
                logger.info(f"Downloaded global model for round {round_number}")
                return model, round_number, True
            else:
                logger.warning(f"No model data in response for round {round_number}")
                return None, round_number, False

        except requests.exceptions.RequestException as e:
            logger.warning(f"Error downloading global model (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                logger.error("Max retries exceeded. Unable to download global model.")
                return None, -1, False


def deserialize_model(serialized_params):
    """
    Deserialize model parameters into a model object.
    Now defaults to SGD with GPU acceleration when available.
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
                    import torch

                    # Extract SGD-specific parameters
                    sgd_params = {
                        'loss': serialized_params.get('loss', 'log'),
                        'penalty': serialized_params.get('penalty', 'l2'),
                        'alpha': serialized_params.get('alpha', 0.0001),
                        'max_iter': serialized_params.get('max_iter', 1000),
                        'tol': serialized_params.get('tol', 1e-3),
                        'random_state': serialized_params.get('random_state', 42)
                    }

                    logger.info("Creating PyTorch GPU-accelerated SGD classifier")
                    return PyTorchSGDClassifier(**sgd_params)

                except ImportError:
                    logger.warning("GPU acceleration requested but PyTorch not available. Falling back to CPU SGD.")

            # CPU-based scikit-learn SGD
            from sklearn.linear_model import SGDClassifier
            logger.info("Creating CPU SGD classifier")
            return SGDClassifier(**sgd_params)

        except Exception as e:
            logger.error(f"Error creating SGD model: {e}")
            raise


    # For backward compatibility, still handle other model types
    elif model_type == 'randomforest':
        logger.warning("RandomForest requested but using SGD instead for consistency")
        # Fall back to SGD with default params
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss='log', penalty='l2', random_state=42)

    else:
        # Unknown model type, default to SGD
        logger.warning(f"Unknown model type '{model_type}', defaulting to SGD")
        from sklearn.linear_model import SGDClassifier
        return SGDClassifier(loss='log', penalty='l2', random_state=42)


def upload_model(model, metrics, round_number):
    """
    Upload the local model to the central server.

    Args:
        model: The trained model
        metrics: Training and evaluation metrics
        round_number: The current global round number

    Returns:
        bool: Success flag
    """
    url = f"{CENTRAL_SERVER}/upload"

    try:
        # Extract model parameters for serialization
        model_params = {
            'client_id': CLIENT_ID,
            'round': round_number,
            'sample_count': metrics.get('total_rows', 0),
            'accuracy': metrics.get('accuracy', 0),
            'model_type': 'sgd',  # Currently only SGD is properly supported
        }

        # If SGD model, extract coefficients
        if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
            model_params['coef'] = model.coef_.tolist() if hasattr(model.coef_, 'tolist') else model.coef_
            model_params['intercept'] = model.intercept_.tolist() if hasattr(model.intercept_,
                                                                             'tolist') else model.intercept_

        # Serialize model parameters
        serialized = pickle.dumps(model_params)

        # Upload to server
        logger.info(f"Uploading model for round {round_number}, size: {len(serialized)} bytes")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    url,
                    files={'model': ('model.pkl', serialized, 'application/octet-stream')},
                    data={'client_id': CLIENT_ID, 'round': round_number},
                    timeout=30
                )
                response.raise_for_status()
                logger.info(f"Model successfully uploaded for round {round_number}")
                return True
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error uploading model (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Max retries exceeded. Unable to upload model.")
                    return False
    except Exception as e:
        logger.error(f"Error preparing model for upload: {e}")
        return False


def main():
    """
    Main function to run the federated learning client.
    """
    logger.info(f"Starting client {CLIENT_ID}")

    # Check if the dataset exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset path not found: {DATASET_PATH}")
        sys.exit(1)

    # Log platform information
    platform_info = get_platform_info()
    logger.info(f"Platform: {platform_info['system']} {platform_info['release']}, "
                f"Python: {platform_info['python_version']}, "
                f"CPU: {platform_info['cpu_cores']} cores, "
                f"RAM: {platform_info['memory_total']:.1f} MB")

    if platform_info.get('gpu_available'):
        logger.info(f"GPU available: {platform_info['gpu_name']}")

    # Main federated learning loop
    current_round = -1
    training_complete = False

    while not training_complete:
        try:
            # Check the server status
            status = get_server_status()
            server_round = status.get('current_round', -1)

            if server_round == -1:
                logger.info("Server not yet initialized. Waiting...")
                time.sleep(5)
                continue

            # Check if we need to get a new model
            if server_round > current_round:
                logger.info(f"New round detected: {server_round}. Downloading global model...")

                # Get the global model
                model, round_number, success = get_global_model()

                if success:
                    # Update our current round
                    current_round = round_number

                    # Load the dataset
                    file_path, X_train, y_train, X_test, y_test = load_data_for_batch_processing(
                        DATASET_PATH,
                        DATA_SOURCE,
                        PARQUET_FILE,
                        TARGET_COLUMN
                    )

                    # Train the model in batches
                    logger.info(f"Training model for round {current_round}")
                    train_start = time.time()

                    # Create the batch processor
                    processor = BatchProcessor(file_path, TARGET_COLUMN, BATCH_SIZE)

                    # Get unique classes for partial_fit
                    unique_classes = get_unique_classes(processor)

                    if USE_PARTIAL_FIT:
                        model = train_model_in_batches(
                            model,
                            processor,
                            unique_classes=unique_classes
                        )
                    else:
                        # For models that don't support partial_fit
                        model.fit(X_train, y_train)

                    train_time = time.time() - train_start
                    logger.info(f"Training completed in {train_time:.2f} seconds")

                    # Evaluate the model
                    logger.info("Evaluating trained model")
                    eval_metrics = evaluate_model_in_batches(model, processor)

                    logger.info(f"Evaluation metrics: {eval_metrics}")

                    # Upload the model to the server
                    logger.info(f"Uploading trained model for round {current_round}")
                    upload_success = upload_model(model, current_round, eval_metrics)

                    if upload_success:
                        logger.info(f"Model successfully uploaded for round {current_round}")
                    else:
                        logger.error(f"Failed to upload model for round {current_round}")

                else:
                    logger.warning("Failed to get global model. Retrying...")
                    time.sleep(RETRY_DELAY)

            # Check if training is complete
            if status.get('training_complete', False):
                logger.info("Training is marked as complete by the server. Exiting...")
                training_complete = True
            else:
                # Wait before checking again
                time.sleep(5)

        except Exception as e:
            logger.error(f"Error in federated learning loop: {str(e)}")
            logger.error(traceback.format_exc())
            time.sleep(RETRY_DELAY)

    logger.info("Client shutting down")


if __name__ == "__main__":
    main()
