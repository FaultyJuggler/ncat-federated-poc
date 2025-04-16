import os
import platform
import multiprocessing
import logging
import subprocess

# Configure logging
logger = logging.getLogger("platform_utils")
logging.basicConfig(level=logging.INFO)


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
        'gpu_count': 0,
        'n_estimators': 100,  # Default for RandomForest
        'vectorize': False
    }

    # Detect system memory
    try:
        # Try psutil first
        import psutil
        total_memory = psutil.virtual_memory().total
        memory_gb = total_memory / (1024 ** 3)
    except ImportError:
        # Fallback method if psutil is not available
        if system == 'Linux':
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                mem_total_line = [line for line in meminfo.split('\n') if 'MemTotal' in line][0]
                mem_kb = int(mem_total_line.split()[1])
                memory_gb = mem_kb / (1024 ** 2)
            except:
                memory_gb = 8  # Default assumption
        else:
            memory_gb = 8  # Default assumption

    logger.info(f"Detected system memory: {memory_gb:.1f} GB")

    # Base memory allocation based on system memory
    if memory_gb >= 128:  # For Lambda Vector with 250GB
        # High-memory system
        config['memory_limit'] = '90%'
        config['batch_size'] = 256
        config['n_estimators'] = 300
    elif memory_gb >= 64:
        # High-memory system
        config['memory_limit'] = '80%'
        config['batch_size'] = 128
        config['n_estimators'] = 200
    elif memory_gb >= 16:
        # Medium-memory system
        config['memory_limit'] = '70%'
        config['batch_size'] = 64
        config['n_estimators'] = 150
    else:
        # Low-memory system
        config['memory_limit'] = '60%'
        config['batch_size'] = 32
        config['n_estimators'] = 100

    # Determine CPU cores
    cpu_count = multiprocessing.cpu_count()

    # Improved GPU detection
    gpu_available = False
    gpu_name = None
    gpu_count = 0

    # Check for NVIDIA GPUs
    try:
        # Try nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split('\n')
            gpu_count = len(gpu_lines)
            if gpu_count > 0:
                gpu_name = gpu_lines[0].split(',')[0].strip()
                gpu_available = True
                config['use_gpu'] = True
                config['platform'] = 'cuda'
                config['gpu_count'] = gpu_count
                logger.info(f"CUDA GPUs detected: {gpu_count} ({gpu_name})")
    except Exception as e:
        logger.debug(f"nvidia-smi detection failed: {e}")

    # Try PyTorch if nvidia-smi failed
    if not gpu_available:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                config['use_gpu'] = True
                config['platform'] = 'cuda'
                config['gpu_count'] = gpu_count
                logger.info(f"CUDA GPU detected via PyTorch: {gpu_name}")
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")

    # Try looking for CUDA environment variables
    if not gpu_available and (os.environ.get('CUDA_VISIBLE_DEVICES') or
                              os.environ.get('NVIDIA_VISIBLE_DEVICES')):
        gpu_available = True
        config['use_gpu'] = True
        config['platform'] = 'cuda'
        config['gpu_count'] = 1
        logger.info("CUDA environment detected through environment variables")

    # Apple Silicon detection
    if system == 'Darwin' and (machine == 'arm64' or 'Apple M' in processor):
        config['platform'] = 'apple_silicon'
        config['vectorize'] = True
        logger.info("Apple Silicon detected")

    # Set number of jobs based on platform and available cores
    if config['platform'] == 'apple_silicon':
        # For Apple Silicon, use all cores but one
        config['n_jobs'] = max(1, cpu_count - 1)
    elif config['platform'] == 'cuda':
        # For CUDA systems with GPUs
        if gpu_count > 0:
            # Adjust parameters for GPU usage
            config['n_jobs'] = min(4, cpu_count)  # Limit CPU usage when using GPU
            config['n_estimators'] = 300  # More trees for GPU acceleration
        else:
            # CUDA platform but no usable GPUs
            config['n_jobs'] = max(1, cpu_count - 2)
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
        # In container, we can be less conservative since we have resource limits
        if memory_gb >= 128:  # For Lambda Vector with 250GB in containers
            config['n_jobs'] = max(1, cpu_count - 1)  # Use more cores in high-memory container

    # Log the detected configuration
    logger.info(f"Platform detection complete: {config['platform']}")
    logger.info(f"CPU cores: {cpu_count}, using {config['n_jobs']} for computation")
    logger.info(f"Memory: {memory_gb:.1f} GB, limiting to {config['memory_limit']}")
    if config['use_gpu']:
        logger.info(f"GPU acceleration enabled: {gpu_count} GPUs detected")

    return config


def optimize_rf_params(config):
    """
    Original RandomForest parameter optimizer (for backward compatibility)
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


def optimize_model_params(config):
    """
    Return optimized model parameters based on the platform configuration.
    Can return different model types based on hardware.
    """
    if config['use_gpu'] and config['platform'] == 'cuda':
        # On GPU systems, use GPU-accelerated algorithms instead of RandomForest
        try:
            import xgboost as xgb
            logger.info("Using GPU-accelerated XGBoost instead of RandomForest")
            return {
                'model_type': 'xgboost',
                'params': {
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'n_estimators': min(300, config['n_estimators']),
                    'max_depth': 10,
                    'learning_rate': 0.1,
                    'objective': 'multi:softprob',
                    'random_state': 42,
                    'n_jobs': 1  # XGBoost handles parallelism differently with GPU
                }
            }
        except ImportError:
            logger.warning("XGBoost not available, falling back to RandomForest")

    # Default to RandomForest with optimized parameters
    return {
        'model_type': 'randomforest',
        'params': {
            'n_estimators': config['n_estimators'],
            'n_jobs': config['n_jobs'],
            'random_state': 42,
            'bootstrap': True,
            'max_features': 'sqrt',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        }
    }