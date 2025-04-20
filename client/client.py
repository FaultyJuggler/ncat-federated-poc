import os
import json
import time
import logging
import requests
import numpy as np
import joblib
import json
import pickle  # For in-memory model serialization
import base64
import pandas as pd
import sys
import traceback
import gc
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

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
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "label")

# Batch processing settings
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10000"))  # Number of rows to process at once
USE_PARTIAL_FIT = os.getenv("USE_PARTIAL_FIT", "false").lower() == "true"  # Whether to use partial_fit or not

# Detect platform and get optimized parameters
platform_config = detect_platform()
logger.info(f"Running on detected platform: {platform_config['platform']}")
logger.info(f"GPU enabled: {platform_config['use_gpu']}")

# Training parameters
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds

# Create a PyTorch-based SGD classifier wrapper
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

def log_memory_usage():
    """Log current memory usage"""
    try:
        import psutil
        usage = psutil.virtual_memory()
        logger.info(
            f"Memory usage: {usage.percent}% ({usage.used / 1024 / 1024:.1f}MB used out of {usage.total / 1024 / 1024:.1f}MB)")
    except ImportError:
        logger.warning("psutil not available, skipping memory usage logging")


class BatchProcessor:
    """Class to handle batch processing of Parquet files with improved memory management"""

    def __init__(self, file_path, target_column, batch_size=10000):
        """Initialize the batch processor"""
        self.file_path = file_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.total_rows = 0
        self.feature_cols = []
        self.categorical_cols = []
        self.num_features = 0
        self.has_initialized = False

    def initialize(self):
        """Initialize by reading metadata and preparing for batch processing"""
        logger.info(f"Initializing batch processor for {self.file_path}")
        log_memory_usage()

        try:
            # Use PyArrow to efficiently read metadata
            import pyarrow.parquet as pq

            # Get file metadata
            parquet_file = pq.ParquetFile(self.file_path)
            self.total_rows = parquet_file.metadata.num_rows
            logger.info(f"Total rows in Parquet file: {self.total_rows}")

            # Read a small sample to determine schema and datatypes
            # Only read first 100 rows for schema detection
            sample = next(self._read_parquet_chunks(parquet_file, max_chunks=1, rows_per_chunk=100))

            # Make sure target column exists
            if self.target_column not in sample.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in Parquet file")

            # Identify feature columns and categorical columns
            self.feature_cols = [col for col in sample.columns if col != self.target_column]
            self.categorical_cols = sample[self.feature_cols].select_dtypes(
                include=['object', 'category']).columns.tolist()

            logger.info(
                f"Identified {len(self.feature_cols)} feature columns and {len(self.categorical_cols)} categorical columns")

            # Extract basic information but don't do full one-hot encoding
            self.num_features = len(self.feature_cols)
            if self.categorical_cols:
                logger.info("Using simple label encoding for categorical features to save memory")

            self.has_initialized = True

            # Force cleanup
            del sample
            del parquet_file
            gc.collect()

        except Exception as e:
            logger.error(f"Error initializing batch processor: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_total_rows(self):
        """Return the total number of rows in the file"""
        if not self.has_initialized:
            self.initialize()
        return self.total_rows

    def get_num_features(self):
        """Return the number of features after encoding"""
        if not self.has_initialized:
            self.initialize()
        return self.num_features

    def _read_parquet_chunks(self, parquet_file=None, max_chunks=None, rows_per_chunk=None):
        """Generator to read chunks of a parquet file with controlled memory usage"""
        if parquet_file is None:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.file_path)

        num_row_groups = parquet_file.metadata.num_row_groups
        chunks_read = 0

        for row_group_idx in range(num_row_groups):
            if max_chunks is not None and chunks_read >= max_chunks:
                break

            try:
                # Read a row group
                table = parquet_file.read_row_group(row_group_idx)
                df = table.to_pandas()

                if rows_per_chunk is not None and rows_per_chunk < len(df):
                    # Split into smaller chunks if needed
                    for i in range(0, len(df), rows_per_chunk):
                        chunks_read += 1
                        if max_chunks is not None and chunks_read > max_chunks:
                            break

                        chunk = df.iloc[i:i + rows_per_chunk].copy()
                        yield chunk

                        # Force cleanup of the chunk
                        del chunk
                        gc.collect()
                else:
                    chunks_read += 1
                    yield df

                # Force cleanup of the dataframe and table
                del df
                del table
                gc.collect()

            except Exception as e:
                logger.error(f"Error reading row group {row_group_idx}: {str(e)}")
                continue

    def batch_iterator(self, max_rows=None):
        """Generator to yield batches of data from the Parquet file with optimized memory usage"""
        if not self.has_initialized:
            self.initialize()

        import pyarrow.parquet as pq

        # Determine how many rows to read
        rows_to_read = min(self.total_rows, max_rows) if max_rows else self.total_rows
        logger.info(f"Will read up to {rows_to_read} rows in batches of {self.batch_size}")

        rows_read = 0

        # Use chunk reader instead of loading row groups directly
        for chunk_df in self._read_parquet_chunks(rows_per_chunk=min(50000, self.batch_size * 5)):
            if rows_read >= rows_to_read:
                break

            # Process in batch_size chunks
            for start_idx in range(0, len(chunk_df), self.batch_size):
                if rows_read >= rows_to_read:
                    break

                end_idx = min(start_idx + self.batch_size, len(chunk_df))
                batch_df = chunk_df.iloc[start_idx:end_idx].copy()

                # Process features and target
                X_batch = batch_df[self.feature_cols].copy()
                y_batch = batch_df[self.target_column].copy()

                # Release batch_df memory
                del batch_df

                # Handle categorical variables with simple label encoding
                if self.categorical_cols:
                    for col in self.categorical_cols:
                        # Check if column exists (defensive coding)
                        if col in X_batch.columns:
                            # Simple label encoding uses less memory than one-hot
                            X_batch[col] = pd.factorize(X_batch[col])[0]

                # Convert to numpy arrays
                X_batch_np = X_batch.to_numpy().astype('float32')
                y_batch_np = y_batch.to_numpy()

                # Release pandas memory
                del X_batch
                del y_batch

                rows_read += len(X_batch_np)

                # Force garbage collection before yielding
                gc.collect()

                yield X_batch_np, y_batch_np

                # Force garbage collection after processing
                gc.collect()

            # Explicitly delete chunk_df after processing
            del chunk_df
            gc.collect()

            # Log progress
            logger.info(f"Processed {rows_read}/{rows_to_read} rows")


def batch_iterator_memory_safe(self, max_rows=None):
    """Ultra memory-safe batch iterator that skips categorical encoding completely"""
    if not self.has_initialized:
        self.initialize()

    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(self.file_path)

    # Determine how many rows to read
    rows_to_read = min(self.total_rows, max_rows) if max_rows else self.total_rows
    logger.info(f"Memory-safe mode: Will read up to {rows_to_read} rows in batches of {self.batch_size}")

    rows_read = 0
    row_group_index = 0

    while rows_read < rows_to_read and row_group_index < parquet_file.metadata.num_row_groups:
        try:
            df = parquet_file.read_row_group(row_group_index).to_pandas()
            row_group_index += 1

            # Process in batch_size chunks
            start_idx = 0
            while start_idx < len(df) and rows_read < rows_to_read:
                end_idx = min(start_idx + self.batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]

                # ULTRA SAFE: Skip all categorical processing
                # Just use numerical columns and convert categoricals to simple integers
                X_batch = batch_df.drop(columns=[self.target_column])
                y_batch = batch_df[self.target_column]

                for col in X_batch.select_dtypes(include=['object', 'category']).columns:
                    # Simple label encoding
                    X_batch[col] = pd.factorize(X_batch[col])[0]

                # Convert to numpy arrays
                X_batch = X_batch.to_numpy().astype('float32')
                y_batch = y_batch.to_numpy()

                batch_rows = len(X_batch)
                rows_read += batch_rows

                # Force garbage collection
                gc.collect()

                yield X_batch, y_batch

                start_idx += self.batch_size

            # Clean up to save memory
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"Error processing row group {row_group_index - 1}: {str(e)}")
            row_group_index += 1  # Skip to next row group on error

def get_unique_classes(self):
    """Return all unique classes in the dataset"""
    # Implementation depends on your data structure
    unique_classes = pd.read_parquet(self.file_path, columns=[self.target_column])[self.target_column].unique()
    return unique_classes


def train_model_in_batches(model, batch_processor, max_rows=None):
    """Train model using batches of data to manage memory usage"""
    logger.info(f"Starting batch training with max_rows={max_rows}")

    log_memory_usage()

    logger.info("Using partial_fit for incremental training")

    total_rows_processed = 0
    batch_count = 0
    start_time = time.time()

    # Define classes for binary classification (goodware=0, malware=1)
    all_classes = np.array([0, 1])

    try:
        for X_batch, y_batch in batch_processor.batch_iterator():
            batch_size = X_batch.shape[0]

            if max_rows and total_rows_processed + batch_size > max_rows:
                # Trim the batch to respect max_rows
                rows_to_take = max_rows - total_rows_processed
                X_batch = X_batch[:rows_to_take]
                y_batch = y_batch[:rows_to_take]
                batch_size = rows_to_take

            # On first batch, pass the classes parameter
            if batch_count == 0:
                model.partial_fit(X_batch, y_batch, classes=all_classes)
                logger.info(f"First batch processed with classes: {all_classes}")
            else:
                model.partial_fit(X_batch, y_batch)

            total_rows_processed += batch_size
            batch_count += 1
            logger.info(f"Processed batch: {batch_size} samples, total: {total_rows_processed}")

            if max_rows and total_rows_processed >= max_rows:
                logger.info(f"Reached max_rows limit ({max_rows}), stopping training")
                break

        logger.info(f"Completed batch training, processed {total_rows_processed} samples")
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Processed {total_rows_processed} rows in {batch_count} batches")

        return model
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def evaluate_model_in_batches(model, batch_processor, max_rows=None):
    """Evaluate a model using batches of data"""
    logger.info("Starting batch evaluation...")
    log_memory_usage()

    total_correct = 0
    total_rows = 0
    batch_count = 0
    start_time = time.time()

    y_true_all = []
    y_pred_all = []

    for X_batch, y_batch in batch_processor.batch_iterator(max_rows):
        batch_count += 1
        batch_size = len(X_batch)

        with joblib.parallel_backend('threading', n_jobs=platform_config['n_jobs']):
            y_pred = model.predict(X_batch)

        # Collect true and predicted values for final metrics
        y_true_all.extend(y_batch)
        y_pred_all.extend(y_pred)

        # Update counters
        total_rows += batch_size

        if batch_count % 10 == 0:
            logger.info(f"Evaluated {batch_count} batches, {total_rows} rows so far")

    # Calculate overall metrics
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_true_all, y_pred_all)

    eval_time = time.time() - start_time
    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")
    logger.info(f"Processed {total_rows} rows in {batch_count} batches")
    logger.info(f"Test accuracy: {accuracy:.4f}")
    logger.info("\nClassification Report:\n" + classification_report(y_true_all, y_pred_all))

    return {"accuracy": accuracy}


def load_data_for_batch_processing():
    """Set up batch processing for data"""
    if DATA_SOURCE.lower() == "parquet" and PARQUET_FILE:
        # Initialize batch processor for Parquet file
        batch_processor = BatchProcessor(PARQUET_FILE, TARGET_COLUMN, BATCH_SIZE)
        batch_processor.initialize()
        return batch_processor
    else:
        # For MNIST, load the whole dataset as before
        logger.info("Loading MNIST dataset...")
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

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a simple batch processor for in-memory data
        class InMemoryBatchProcessor:
            def __init__(self, X, y, batch_size):
                self.X = X
                self.y = y
                self.batch_size = batch_size
                self.total_rows = len(X)
                self.num_features = X.shape[1]

            def get_total_rows(self):
                return self.total_rows

            def get_num_features(self):
                return self.num_features

            def batch_iterator(self, max_rows=None):
                rows_to_process = min(self.total_rows, max_rows) if max_rows else self.total_rows
                for i in range(0, rows_to_process, self.batch_size):
                    end_idx = min(i + self.batch_size, rows_to_process)
                    yield self.X[i:end_idx], self.y[i:end_idx]

        # Create train and test batch processors
        train_processor = InMemoryBatchProcessor(X_train, y_train, BATCH_SIZE)
        test_processor = InMemoryBatchProcessor(X_test, y_test, BATCH_SIZE)

        logger.info(
            f"Using in-memory batch processing for MNIST with {len(X_train)} training and {len(X_test)} test examples")
        return train_processor, test_processor


def create_model():
    """Create a model optimized for the current platform"""
    model_config = optimize_model_params(platform_config)

    if model_config['model_type'] == 'sgd':
        from sklearn.linear_model import SGDClassifier
        logger.info("Creating SGDClassifier model")
        return SGDClassifier(**model_config['params'])
    else:
        # Default to RandomForest
        from sklearn.ensemble import RandomForestClassifier
        logger.info(f"Creating RandomForest with optimized parameters")
        return RandomForestClassifier(**model_config['params'])


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



def upload_model(model, sample_count, metrics):
    try:
        # Serialize the model to a binary string using pickle
        model_bytes = pickle.dumps(model)

        # Base64 encode the binary data to make it JSON-safe
        model_encoded = base64.b64encode(model_bytes).decode('utf-8')

        # Create the metadata payload
        metadata = {
            'client_id': CLIENT_ID,
            'sample_count': sample_count,
            'metrics': metrics,  # Optional: Add relevant metrics
            'model': model_encoded  # Send the serialized model
        }
        logger.info(f"Serialized model into metadata payload: {type(metadata)}")

        # Send the metadata + serialized model as JSON
        response = requests.post(f"{CENTRAL_SERVER}/upload", json=metadata)

        # Handle response
        if response.status_code == 200:
            logger.info(f"Successfully uploaded model metadata. Response: {response.text}")
            return True
        else:
            logger.warning(f"Failed to upload: {response.status_code}, {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error while uploading model: {str(e)}")
        return False


def main():
    """Main function to run the client in the federated learning process"""
    # Log GPU status
    if platform_config['use_gpu']:
        gpu_count = platform_config.get('gpu_count', os.environ.get('GPU_COUNT', 1))
        logger.info(f"GPU acceleration enabled: {gpu_count} GPUs detected")

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
        # Load data for this client - now returns batch processors
        if DATA_SOURCE.lower() == "parquet" and PARQUET_FILE:
            train_processor = load_data_for_batch_processing()
            # For parquet, we'll create a separate processor for testing
            test_processor = BatchProcessor(PARQUET_FILE, TARGET_COLUMN, BATCH_SIZE)
            test_processor.initialize()

            # Log info about the dataset
            logger.info(
                f"Parquet file contains {train_processor.get_total_rows()} rows and {train_processor.get_num_features()} features")

        else:
            # For MNIST, we get separate train and test processors
            train_processor, test_processor = load_data_for_batch_processing()
            logger.info(
                f"MNIST dataset: {train_processor.get_total_rows()} training rows and {test_processor.get_total_rows()} test rows")

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

            # Train the model using batches
            try:
                # For large datasets, let's limit how much we train on
                # This is optional and can be adjusted based on your needs
                max_train_rows = 500000  # Limit training to 500K rows

                model = train_model_in_batches(model, train_processor, max_train_rows)
            except Exception as e:
                logger.error(f"Error training model: {e}")
                logger.error(traceback.format_exc())
                continue

            # Evaluate the model using batches
            try:
                # For evaluation, we can use a smaller subset
                max_eval_rows = 100000  # Limit evaluation to 100K rows

                metrics = evaluate_model_in_batches(model, test_processor, max_eval_rows)
            except Exception as e:
                logger.error(f"Error evaluating model: {e}")
                metrics = {"accuracy": 0.0}

            # Save model checkpoint
            checkpoint_path = f"{DATASET_PATH}/models/model_{CLIENT_ID}_round_{current_round}.joblib"
            joblib.dump(model, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Upload the trained model
            try:
                # Use the total processed rows for sample count
                sample_count = min(train_processor.get_total_rows(), max_train_rows)

                response = upload_model(model, sample_count, metrics)
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