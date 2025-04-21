import os
import platform
import multiprocessing
import psutil
import logging

# Configure logging
logger = logging.getLogger("platform_utils")


def detect_platform():
    """
    Detect the platform and return configuration parameters
    optimized for the current environment.

    Returns:
        dict: A dictionary containing platform-specific configurations
    """
    system = platform.system()
    processor = platform.processor()
    machine = platform.machine()

    # Default configuration
    config = {
        'platform': 'unknown',
        'n_jobs': 1,  # Default to single core
        'memory_limit': None,  # No explicit memory limit
        'batch_size': 32,
        'use_gpu': False,
        'n_estimators': 100,  # Default for RandomForest
        'vectorize': False
    }

    # Detect system memory
    total_memory = psutil.virtual_memory().total
    memory_gb = total_memory / (1024 ** 3)

    # Base memory allocation based on system memory
    if memory_gb >= 64:
        # High-memory system
        config['memory_limit'] = '80%'
        config['batch_size'] = 128
    elif memory_gb >= 16:
        # Medium-memory system
        config['memory_limit'] = '70%'
        config['batch_size'] = 64
    else:
        # Low-memory system
        config['memory_limit'] = '60%'
        config['batch_size'] = 32

    # Determine CPU cores
    cpu_count = multiprocessing.cpu_count()

    # CUDA/GPU detection
    gpu_available = False
    gpu_name = None

    try:
        # Try detecting CUDA via PyTorch
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            config['use_gpu'] = True
            config['platform'] = 'cuda'
            logger.info(f"CUDA GPU detected: {gpu_name}")
    except (ImportError, Exception):
        # PyTorch not available, try alternative methods
        pass

    if not gpu_available:
        try:
            # Try detecting CUDA through environment variables
            if os.environ.get('CUDA_VISIBLE_DEVICES') or os.environ.get('NVIDIA_VISIBLE_DEVICES'):
                gpu_available = True
                config['use_gpu'] = True
                config['platform'] = 'cuda'
                logger.info("CUDA environment detected through environment variables")
        except Exception:
            pass

    # Apple Silicon detection
    if system == 'Darwin' and (machine == 'arm64' or 'Apple M' in processor):
        config['platform'] = 'apple_silicon'
        config['vectorize'] = True
        logger.info("Apple Silicon detected")

        # Additional optimizations for Apple Silicon
        try:
            import sklearn
            # Check if we have scikit-learn 1.0+ which has better M1 optimizations
            if int(sklearn.__version__.split('.')[0]) >= 1:
                config['vectorize'] = True
        except (ImportError, Exception):
            pass

    # Set number of jobs based on platform and available cores
    if config['platform'] == 'apple_silicon':
        # For Apple Silicon, use all cores but one
        config['n_jobs'] = max(1, cpu_count - 1)
        # Optimize number of estimators
        config['n_estimators'] = 150  # More trees for better quality
    elif config['platform'] == 'cuda':
        # For CUDA systems, let GPU do heavier work
        config['n_jobs'] = min(4, cpu_count)  # Limit for GPU systems to avoid CPU bottleneck
        config['n_estimators'] = 200  # More trees for GPU
    else:
        # For standard systems, use a balanced approach
        config['n_jobs'] = max(1, cpu_count - 2)

    # Docker container detection
    in_container = False
    if os.path.exists('/.dockerenv'):
        in_container = True
    elif os.environ.get('DOCKER_CONTAINER', False):
        in_container = True

    if in_container:
        logger.info("Running inside a Docker container")
        # In container, be more conservative with resources
        config['n_jobs'] = max(1, config['n_jobs'] - 1)
        config['memory_limit'] = '80%'  # Lower memory limit in containers

    # Log the detected configuration
    logger.info(f"Platform detection complete: {config['platform']}")
    logger.info(f"CPU cores: {cpu_count}, using {config['n_jobs']} for computation")
    logger.info(f"Memory: {memory_gb:.1f} GB, limiting to {config['memory_limit']}")
    if config['use_gpu']:
        logger.info(f"GPU acceleration enabled: {gpu_name}")

    return config

# SGD classifier
def optimize_model_params(config):
    """Return optimized model parameters based on the platform configuration."""

    # Determine model type based on available libraries
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False


    # Use SGDClassifier instead of XGBoost
    from sklearn.linear_model import SGDClassifier
    logger.info("Using memory-efficient SGDClassifier for federated learning")

    # Generate params for sklearn SGD
    params = {
        'model_type': 'pytorch_sgd',
        'params': {
            'loss': 'log_loss',  # For classification
            'penalty': 'l2',
            'alpha': 0.0001,
            'max_iter': 5,
            'tol': 0.001,
            'random_state': 42,
            'warm_start': True  # Important for incremental learning
        }
    }

    # Use PyTorch if available
    if has_torch:
    # Remove sklearn-specific params
        if 'warm_start' in params['params']:
            del params['params']["warm_start"]

    return params


def optimize_rf_params(config):
    """
    Return optimized RandomForest parameters based on the platform configuration.

    Args:
        config: Platform configuration from detect_platform()

    Returns:
        dict: Parameters for RandomForestClassifier optimized for the platform
    """
    params = {
        'n_estimators': config['n_estimators'],
        'n_jobs': config['n_jobs'],
        'random_state': 42
    }

    # Platform-specific optimizations
    if config['platform'] == 'apple_silicon':
        # Optimize for Apple Silicon
        params.update({
            'bootstrap': True,
            'max_features': 'sqrt',  # For better vectorization
            'min_samples_leaf': 2,  # Slightly more conservative to reduce overfitting
        })
    elif config['platform'] == 'cuda':
        # Optimize for CUDA systems
        params.update({
            'bootstrap': True,
            'max_features': 'sqrt',
            'min_samples_split': 5,  # For better parallelization
            'min_samples_leaf': 1,  # Default value
        })
    else:
        # General optimizations for standard systems
        params.update({
            'bootstrap': True,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        })

    return params
