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
    """Class to handle batch processing of Parquet files"""

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
            sample = parquet_file.read_row_group(0).to_pandas()

            # Make sure target column exists
            if self.target_column not in sample.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in Parquet file")

            # Identify feature columns and categorical columns
            self.feature_cols = [col for col in sample.columns if col != self.target_column]
            self.categorical_cols = sample[self.feature_cols].select_dtypes(
                include=['object', 'category']).columns.tolist()

            logger.info(
                f"Identified {len(self.feature_cols)} feature columns and {len(self.categorical_cols)} categorical columns")

            # Handle a small sample to determine one-hot encoding columns
            if self.categorical_cols:
                # Convert categorical to one-hot
                sample_x = pd.get_dummies(sample[self.feature_cols], columns=self.categorical_cols, drop_first=True)
                self.num_features = sample_x.shape[1]
                logger.info(f"After one-hot encoding: {self.num_features} features")
            else:
                self.num_features = len(self.feature_cols)

            self.has_initialized = True

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
        """Return the number of features after one-hot encoding"""
        if not self.has_initialized:
            self.initialize()
        return self.num_features

    def batch_iterator(self, max_rows=None):
        """Generator to yield batches of data from the Parquet file"""
        if not self.has_initialized:
            self.initialize()

        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(self.file_path)

        # Determine how many rows to read
        rows_to_read = min(self.total_rows, max_rows) if max_rows else self.total_rows
        logger.info(f"Will read up to {rows_to_read} rows in batches of {self.batch_size}")

        rows_read = 0
        row_group_index = 0
        df_remainder = None

        while rows_read < rows_to_read and row_group_index < parquet_file.metadata.num_row_groups:
            # Read a row group
            try:
                df = parquet_file.read_row_group(row_group_index).to_pandas()
                row_group_index += 1

                # Append any remaining rows from previous batch
                if df_remainder is not None and len(df_remainder) > 0:
                    df = pd.concat([df_remainder, df], ignore_index=True)
                    df_remainder = None

                # Process in batch_size chunks
                start_idx = 0
                while start_idx < len(df) and rows_read < rows_to_read:
                    end_idx = min(start_idx + self.batch_size, len(df))
                    batch_df = df.iloc[start_idx:end_idx]

                    # Process features and target
                    X_batch = batch_df[self.feature_cols]
                    y_batch = batch_df[self.target_column]

                    # Handle categorical variables
                    if self.categorical_cols:
                        X_batch = pd.get_dummies(X_batch, columns=self.categorical_cols, drop_first=True)

                    # Convert to numpy arrays
                    X_batch = X_batch.to_numpy().astype('float32')
                    y_batch = y_batch.to_numpy()

                    batch_rows = len(X_batch)
                    rows_read += batch_rows

                    yield X_batch, y_batch

                    start_idx += self.batch_size

                # If there's a remainder from this chunk, save it for the next iteration
                if start_idx < len(df):
                    df_remainder = df.iloc[start_idx:]

                # Clean up to save memory
                del df

            except Exception as e:
                logger.error(f"Error processing row group {row_group_index - 1}: {str(e)}")
                logger.error(traceback.format_exc())
                row_group_index += 1  # Skip to next row group on error

        logger.info(f"Completed reading {rows_read} rows from Parquet file")


def train_model_in_batches(model, batch_processor, max_rows=None):
    """Train a model using batches of data"""
    logger.info("Starting batch training...")
    log_memory_usage()

    total_rows_processed = 0
    batch_count = 0
    start_time = time.time()

    # For XGBoost with GPU, we need to collect data first
    using_xgboost_gpu = (hasattr(model, 'tree_method') and
                         getattr(model, 'tree_method', '') == 'gpu_hist')

    if using_xgboost_gpu and not USE_PARTIAL_FIT:
        logger.info("Using XGBoost with GPU - collecting all data for training")
        # For XGBoost with GPU, we collect all data first
        all_X = []
        all_y = []

        for X_batch, y_batch in batch_processor.batch_iterator(max_rows):
            all_X.append(X_batch)
            all_y.append(y_batch)
            batch_count += 1
            total_rows_processed += len(X_batch)

            if batch_count % 10 == 0:
                logger.info(f"Loaded {batch_count} batches, {total_rows_processed} rows so far")
                log_memory_usage()

        # Combine all batches
        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)

        logger.info(f"Training XGBoost with {len(X_train)} rows and {X_train.shape[1]} features")
        model.fit(X_train, y_train)

    else:
        # For other models or if using partial_fit, process in batches
        # Check if model has partial_fit method or we need standard fit
        has_partial_fit = hasattr(model, 'partial_fit') and USE_PARTIAL_FIT

        # If model doesn't support partial_fit and we're not forcing standard fit,
        # we'll collect the data and use a single fit call
        if not has_partial_fit:
            logger.info("Model doesn't support partial_fit, collecting all data for fit")
            all_X = []
            all_y = []

            for X_batch, y_batch in batch_processor.batch_iterator(max_rows):
                all_X.append(X_batch)
                all_y.append(y_batch)
                batch_count += 1
                total_rows_processed += len(X_batch)

                if batch_count % 10 == 0:
                    logger.info(f"Loaded {batch_count} batches, {total_rows_processed} rows so far")
                    log_memory_usage()

            # Combine all batches
            X_train = np.vstack(all_X)
            y_train = np.concatenate(all_y)

            logger.info(f"Training model with {len(X_train)} rows and {X_train.shape[1]} features")

            # Use parallel backend if supported
            if hasattr(model, 'n_jobs'):
                with joblib.parallel_backend('threading', n_jobs=platform_config['n_jobs']):
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

        else:
            # For models with partial_fit, train incrementally
            logger.info("Using partial_fit for incremental training")

            # Get list of classes for supervised classification
            classes = None
            if hasattr(model, 'classes_'):
                classes = model.classes_

            for X_batch, y_batch in batch_processor.batch_iterator(max_rows):
                batch_count += 1
                batch_size = len(X_batch)

                # Train on this batch
                if classes is not None and batch_count == 1:
                    # For first batch, we need to provide classes
                    model.partial_fit(X_batch, y_batch, classes=np.unique(y_batch))
                else:
                    model.partial_fit(X_batch, y_batch)

                total_rows_processed += batch_size

                if batch_count % 5 == 0:
                    logger.info(f"Processed {batch_count} batches, {total_rows_processed} rows so far")
                    log_memory_usage()

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Processed {total_rows_processed} rows in {batch_count} batches")
    log_memory_usage()

    return model


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

        # Make predictions on this batch
        if hasattr(model, 'tree_method') and getattr(model, 'tree_method', '') == 'gpu_hist':
            # XGBoost with GPU
            y_pred = model.predict(X_batch)
        else:
            # Standard model
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