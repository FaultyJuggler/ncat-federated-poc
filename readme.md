# Federated Learning with RandomForest

This project implements a federated learning system using RandomForest classifiers. The system consists of a central server and multiple client nodes that train on local data and collaborate to build a global model without sharing raw data.

## Overview

The implementation features:

- Docker-based architecture with separate containers for central server and clients
- RandomForest machine learning models (scikit-learn based)
- Support for both MNIST data and custom Parquet files
- Federated model aggregation through tree sampling
- REST API for model exchange
- Visualization tools for monitoring training progress

## Project Structure

```
federated_learning_project/
├── central_server/
│   ├── model.py             # Central model and aggregation logic
│   ├── server.py            # Central coordination server
│   ├── visualize.py         # Metrics visualization tool
│   └── requirements.txt     # Dependencies for central server
├── client/
│   ├── client.py            # Client implementation
│   └── requirements.txt     # Dependencies for clients
├── docker/
│   ├── central/Dockerfile   # Central server container
│   └── client/Dockerfile    # Client container
├── tools/
│   └── generate_parquet_samples.py  # Helper for creating test data
├── docker-compose.yml       # Multi-container configuration
└── README.md                # Documentation
```

## Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local tools)
- pandas, pyarrow, and scikit-learn (for data generation tools)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd federated_learning_project
   ```

2. Create data directories:
   ```bash
   mkdir -p data/client1 data/client2 data/client3
   ```

3. If using Parquet data, generate sample files or prepare your own:
   ```bash
   python tools/generate_parquet_samples.py
   ```

## Configuration

### Docker Compose

The `docker-compose.yml` file defines the central server and client containers. Key configuration options include:

- `DATA_SOURCE`: Set to either "mnist" (default) or "parquet"
- `PARQUET_FILE`: Path to Parquet file (when using Parquet data source)
- `TARGET_COLUMN`: Column name for prediction target in Parquet file

Example for using Parquet data:

```yaml
client1:
  environment:
    - CLIENT_ID=client1
    - CENTRAL_SERVER=http://central:8080
    - DATASET_PATH=/data/dataset1
    - DATA_SOURCE=parquet
    - PARQUET_FILE=/data/dataset1.parquet
    - TARGET_COLUMN=target
```

## Running the System

1. Start the federated learning system:
   ```bash
   docker compose up --build
   ```

2. Monitor the training process through logs:
   ```bash
   docker compose logs -f
   ```

3. Visualize training metrics (from host machine):
   ```bash
   python central_server/visualize.py
   ```

## How It Works

### Federated Learning Process

1. **Initialization**: Central server starts and waits for clients to connect.
2. **Model Distribution**: Clients download the initial global model configuration.
3. **Local Training**: Each client trains a RandomForest model on its local dataset.
4. **Model Uploading**: Clients send their trained models to the central server.
5. **Model Aggregation**: The server creates a new global model by taking a weighted subset of trees from each client model.
6. **Repeat**: Steps 2-5 are repeated for multiple rounds until convergence or a fixed number of rounds.

### RandomForest Aggregation

The federated RandomForest implementation uses a tree sampling strategy:

1. Each client trains a complete RandomForest model locally.
2. The central server selects trees from each client model proportional to their dataset sizes.
3. These trees are combined to form a new global ensemble model.
4. Configuration parameters (but not trees) are distributed back to clients for the next round.

## Using Your Own Data

To use your own data in Parquet format:

1. Prepare Parquet files with features and a target column.
2. Place the files in the appropriate client data directories.
3. Configure the `docker-compose.yml` file with:
   - `DATA_SOURCE=parquet`
   - Correct path to your Parquet file
   - Name of the target column

## Monitoring and Visualization

The `visualize.py` script provides real-time monitoring of the federated learning process:

- Displays accuracy metrics over training rounds
- Automatically updates as training progresses
- Saves plots to disk for later reference

## Customization

### Model Parameters

To customize the RandomForest parameters, modify the `create_model()` function in both `central_server/model.py` and `client/client.py`:

```python
def create_model():
    return RandomForestClassifier(
        n_estimators=200,  # Increase number of trees
        max_depth=10,      # Limit tree depth
        min_samples_split=5,
        # Other parameters...
    )
```

### Data Partitioning

By default, the system splits data evenly among clients. To change this behavior, modify the data partitioning logic in `client/client.py`.

## Troubleshooting

### Connection Issues

If clients cannot connect to the central server:
- Ensure the `CENTRAL_SERVER` environment variable matches the service name in `docker-compose.yml`
- Check that port 8080 is not being used by another application

### Memory Issues

If you encounter memory errors when using large Parquet files:
- Increase Docker container memory limits
- Consider subsampling your data or reducing the size of the RandomForest (fewer trees)

### Training Failures

If model training fails:
- Check logs for specific error messages
- Ensure the target column is properly formatted (classification requires integer labels)
- Verify that Parquet files have consistent schema across clients

## Advanced Usage

### Regression Tasks

To use the system for regression instead of classification:
1. Replace `RandomForestClassifier` with `RandomForestRegressor`
2. Update evaluation metrics in `evaluate_model()` to use regression metrics like MSE

### Production Deployment

For production use:
- Implement user authentication for the central server
- Add secure HTTPS connections
- Consider implementing differential privacy techniques
- Add monitoring and logging systems

### Additional Tools

- `partition_large_parquet.py`: Splits a large Parquet file into smaller chunks for distributed training. Now includes enhanced handling for DataFrame schema consistency, threading-based force-exit timer for graceful shutdown, and improved progress tracking.

   ### Partitioning Large Datasets

   If you have a large dataset in Parquet format and need to partition it for the clients, you can use the `partition_large_parquet.py` tool. 

   ```bash
   cd tools
   pip install -r requirements.txt
   python partition_large_parquet.py --input_file <path_to_parquet_file> --output_dir ../data --method row
   ```

   Key Features:
   - **Graceful Shutdown Handling**: The tool now listens for interrupts (Ctrl+C) and cleans up resources gracefully. If shutdown times out (after 2.5 minutes), it forces an exit.
   - **Schema Validation**: Automatically checks and aligns DataFrame schemas with the Parquet file schema. Missing columns are filled with default values to ensure consistency.
   - **Multithreaded Shutdown Timer**: Implements a timeout mechanism for shutdown, ensuring processing doesn't hang indefinitely.
   - **Optimized Performance**: Uses an adaptive batch size and multiprocessing for efficient large-scale data partitioning.

   #### Example Output Directory Structure
   After running the tool, the output will look like:
```
data/
├── client1/
│ └── dataset.parquet
├── client2/
│ └── dataset.parquet
└── client3/
└── dataset.parquet
```
- **Force Exit During Shutdown**: If the tool cannot complete processing within 2.5 minutes after receiving a shutdown signal, it will forcefully exit and log a timeout message. Ensure your dataset and output directory permissions are set correctly to avoid hangs.

## License

[Specify your license information here]

## Acknowledgments

This project uses the following open-source packages:
- scikit-learn
- pandas
- pyarrow
- fastparquet
- Flask
- Docker
