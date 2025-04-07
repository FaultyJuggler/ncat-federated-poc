import os
import sys
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import platform
from tqdm import tqdm
import multiprocessing
import psutil
import gc


def detect_platform():
    """Detect the running platform and return optimized parameters"""
    system = platform.system()
    processor = platform.processor()
    machine = platform.machine()

    # Default parameters
    params = {
        'is_apple_silicon': False,
        'is_cuda_available': False,
        'optimal_processes': max(1, multiprocessing.cpu_count() - 1),
        'memory_factor': 0.25,  # Use 25% of available memory by default
        'preferable_method': 'row'  # Default method
    }

    print(f"Detected platform: {system} on {processor} ({machine})")

    # Apple Silicon detection
    if system == 'Darwin' and (machine == 'arm64' or 'Apple M' in processor):
        params['is_apple_silicon'] = True
        params['memory_factor'] = 0.3  # Apple Silicon has unified memory, can use more
        params['preferable_method'] = 'random'  # Random partitioning works well with multiprocessing on M-series
        print("Detected Apple Silicon (M-series) processor")

    # CUDA detection for NVIDIA GPUs
    if system == 'Linux' or system == 'Windows':
        try:
            import cupy
            params['is_cuda_available'] = True
            print("CUDA support detected")
        except (ImportError, ModuleNotFoundError):
            try:
                import torch
                params['is_cuda_available'] = torch.cuda.is_available()
                if params['is_cuda_available']:
                    print(f"CUDA support detected through PyTorch: {torch.cuda.get_device_name(0)}")
            except (ImportError, ModuleNotFoundError):
                # Neither CuPy nor PyTorch available
                pass

    # Memory optimization based on platform
    total_memory = psutil.virtual_memory().total
    if total_memory >= 32 * (1024 ** 3):  # 32GB or more RAM
        params['memory_factor'] = 0.4
        print(f"High memory system detected ({total_memory / (1024 ** 3):.1f} GB)")
    elif total_memory <= 8 * (1024 ** 3):  # 8GB or less RAM
        params['memory_factor'] = 0.2
        print(f"Limited memory system detected ({total_memory / (1024 ** 3):.1f} GB)")

    # CPU cores optimization
    if params['optimal_processes'] >= 8:
        print(f"Multi-core system detected with {multiprocessing.cpu_count()} cores")
        # For high core count systems, leave more cores for system processes
        params['optimal_processes'] = max(4, multiprocessing.cpu_count() - 2)

    return params


def get_optimal_batch_size(memory_factor=0.25):
    """Determine an optimal batch size based on available memory"""
    # Get available memory in bytes
    available_memory = psutil.virtual_memory().available

    # Use specified portion of available memory
    usable_memory = available_memory * memory_factor

    # Assume approximately 1000 bytes per row (adjust based on your data)
    bytes_per_row = 1000

    # Calculate batch size that would use memory efficiently
    optimal_batch_size = int(usable_memory / bytes_per_row)

    # Cap at a reasonable maximum and minimum
    return max(10000, min(optimal_batch_size, 500000))


def process_batch(args):
    """Process a single batch of data - designed for multiprocessing"""
    batch_df, client_id, output_file, schema_dict = args

    if not batch_df.empty:
        # Recreate the schema - we need to do this because schema can't be pickled
        schema = pa.schema(schema_dict['fields'])

        # Write to client's file in append mode if it exists
        table = pa.Table.from_pandas(batch_df)
        if os.path.exists(output_file):
            with pq.ParquetWriter(output_file, schema, append=True) as writer:
                writer.write_table(table)
        else:
            with pq.ParquetWriter(output_file, schema) as writer:
                writer.write_table(table)

    return len(batch_df)


def partition_large_parquet(input_file, output_dir, method=None, partition_column=None,
                            batch_size=None, num_processes=None):
    """
    Partition a large Parquet file into three parts for federated learning clients.
    Automatically optimizes for the current platform.

    Args:
        input_file: Path to the input Parquet file
        output_dir: Directory to save the partitioned files
        method: Partitioning method ('row', 'column', or 'random')
        partition_column: Column name for value-based partitioning
        batch_size: Number of rows to process at once (auto-calculated if None)
        num_processes: Number of processes to use (auto-calculated if None)
    """
    # Detect platform and get optimized parameters
    platform_params = detect_platform()

    # Set method based on platform if not specified
    if method is None:
        method = platform_params['preferable_method']
        print(f"Auto-selected partitioning method: {method}")

    # Auto-determine batch size if not provided
    if batch_size is None:
        batch_size = get_optimal_batch_size(platform_params['memory_factor'])
        print(f"Auto-determined batch size: {batch_size} rows")

    # Determine optimal number of processes if not provided
    if num_processes is None:
        num_processes = platform_params['optimal_processes']
        print(f"Using {num_processes} processes for parallel operations")

    # Create output directories if they don't exist
    for i in range(1, 4):
        client_dir = f"{output_dir}/client{i}"
        os.makedirs(client_dir, exist_ok=True)
        # Remove existing output files to avoid append issues
        output_file = f"{client_dir}/dataset.parquet"
        if os.path.exists(output_file):
            os.remove(output_file)

    print(f"Processing large Parquet file: {input_file}")

    # Inspect the Parquet file metadata without loading it fully
    parquet_file = pq.ParquetFile(input_file)
    num_row_groups = parquet_file.metadata.num_row_groups
    total_rows = parquet_file.metadata.num_rows

    # Get the Arrow schema from the Parquet schema
    arrow_schema = parquet_file.schema_arrow

    # Convert schema to dictionary representation for multiprocessing
    schema_dict = {
        'fields': [(field.name, field.type) for field in arrow_schema]
    }

    print(f"Total rows: {total_rows}")
    print(f"Number of row groups: {num_row_groups}")
    print(f"Schema fields: {[f.name for f in arrow_schema]}")

    # Method 1: Row-based partitioning
    if method == 'row':
        print("Using row-based partitioning with PyArrow")

        # Calculate row ranges for each client
        rows_per_client = total_rows // 3
        client_ranges = [
            (0, rows_per_client),
            (rows_per_client, 2 * rows_per_client),
            (2 * rows_per_client, total_rows)
        ]

        # Process each client's data range
        for client_id, (start_row, end_row) in enumerate(client_ranges, 1):
            print(f"Processing Client {client_id}: rows {start_row} to {end_row}")

            # Create a writer for this client
            client_file = f"{output_dir}/client{client_id}/dataset.parquet"
            writer = None
            rows_written = 0

            # Read and process data in batches
            current_row = 0
            for batch_id in range(num_row_groups):
                # Read one row group at a time
                batch = parquet_file.read_row_group(batch_id)
                batch_df = batch.to_pandas()
                batch_size = len(batch_df)

                # Check if this batch overlaps with the client's range
                if current_row + batch_size <= start_row:
                    # This batch is before the client's range, skip it
                    current_row += batch_size
                    continue

                if current_row >= end_row:
                    # We've processed all rows for this client
                    break

                # Calculate the portion of this batch that belongs to the client
                batch_start = max(0, start_row - current_row)
                batch_end = min(batch_size, end_row - current_row)

                if batch_start < batch_size and batch_end > 0:
                    # Extract the relevant portion of the batch
                    client_batch = batch_df.iloc[batch_start:batch_end]

                    # Write to client's file
                    if writer is None:
                        writer = pq.ParquetWriter(client_file, arrow_schema)

                    writer.write_table(pa.Table.from_pandas(client_batch))
                    rows_written += len(client_batch)

                current_row += batch_size

                # Force garbage collection
                del batch_df
                gc.collect()

            if writer:
                writer.close()
            print(f"Client {client_id}: {rows_written} rows written to {client_file}")

    # Method 2: Column-value based partitioning
    elif method == 'column' and partition_column is not None:
        print(f"Using value-based partitioning on column: {partition_column}")

        # First, scan the file to get unique values of the partition column
        dataset = ds.dataset(input_file, format='parquet')

        # Check that partition column exists
        if partition_column not in arrow_schema.names:
            raise ValueError(
                f"Column '{partition_column}' not found in dataset. Available columns: {arrow_schema.names}")

        scanner = dataset.scanner(columns=[partition_column])
        unique_values = set()

        print("Scanning for unique values...")
        for batch in tqdm(scanner.to_batches(), total=num_row_groups):
            batch_values = batch[partition_column].to_numpy()
            unique_values.update(set(batch_values))
            del batch_values
            gc.collect()

        unique_values = list(unique_values)
        np.random.shuffle(unique_values)  # Randomize to avoid potential biases

        print(f"Found {len(unique_values)} unique values for {partition_column}")

        # Split unique values into 3 groups
        values_per_client = len(unique_values) // 3
        client_values = [
            set(unique_values[:values_per_client]),
            set(unique_values[values_per_client:2 * values_per_client]),
            set(unique_values[2 * values_per_client:])
        ]

        # Client output files
        client_files = [f"{output_dir}/client{i + 1}/dataset.parquet" for i in range(3)]
        client_counters = [0, 0, 0]

        # Process the file in batches and route rows to the appropriate client
        print("Processing and routing data...")

        # Use multiprocessing if appropriate for the platform
        use_multiprocessing = platform_params['is_apple_silicon'] or num_processes > 1
        if use_multiprocessing:
            # Create a multiprocessing pool for parallel processing
            with multiprocessing.Pool(processes=num_processes) as pool:
                for batch_id in tqdm(range(num_row_groups)):
                    # Read one row group at a time
                    batch = parquet_file.read_row_group(batch_id)
                    batch_df = batch.to_pandas()

                    # Prepare batch processing tasks
                    tasks = []

                    # Split the batch based on partition column values
                    for client_id in range(3):
                        # Filter rows for this client
                        client_mask = batch_df[partition_column].isin(client_values[client_id])
                        client_batch = batch_df[client_mask].copy()

                        # Only process non-empty batches
                        if not client_batch.empty:
                            tasks.append((client_batch, client_id, client_files[client_id], schema_dict))

                    # Process batches in parallel
                    if tasks:
                        results = pool.map(process_batch, tasks)

                        # Update counters
                        for i, result in enumerate(results):
                            if i < len(client_counters):
                                client_counters[i] += result

                    # Force garbage collection
                    del batch_df
                    gc.collect()
        else:
            # Sequential processing for systems where multiprocessing might be less efficient
            for batch_id in tqdm(range(num_row_groups)):
                # Read one row group at a time
                batch = parquet_file.read_row_group(batch_id)
                batch_df = batch.to_pandas()

                # Split the batch based on partition column values
                for client_id in range(3):
                    # Filter rows for this client
                    client_mask = batch_df[partition_column].isin(client_values[client_id])
                    client_batch = batch_df[client_mask]

                    if not client_batch.empty:
                        # Process this batch
                        rows_added = process_batch((client_batch, client_id, client_files[client_id], schema_dict))
                        client_counters[client_id] += rows_added

                # Force garbage collection
                del batch_df
                gc.collect()

        for i in range(3):
            print(f"Client {i + 1}: {client_counters[i]} rows written")

    # Method 3: Random partitioning
    elif method == 'random':
        print("Using memory-efficient random partitioning")

        # Client output files
        client_files = [f"{output_dir}/client{i + 1}/dataset.parquet" for i in range(3)]
        client_counters = [0, 0, 0]

        # Set random seed for reproducibility
        np.random.seed(42)

        # Process the file in batches
        print("Processing and randomly assigning data...")

        # Use multiprocessing if appropriate for the platform
        use_multiprocessing = platform_params['is_apple_silicon'] or num_processes > 1
        if use_multiprocessing:
            # Create a multiprocessing pool for parallel processing
            with multiprocessing.Pool(processes=num_processes) as pool:
                for batch_id in tqdm(range(num_row_groups)):
                    try:
                        # Read one row group at a time
                        batch = parquet_file.read_row_group(batch_id)
                        batch_df = batch.to_pandas()
                        batch_size = len(batch_df)

                        # Generate random assignments for this batch
                        assignments = np.random.choice([0, 1, 2], size=batch_size)

                        # Prepare batch processing tasks
                        tasks = []

                        # Split the batch based on assignments
                        for client_id in range(3):
                            # Filter rows for this client
                            client_batch = batch_df[assignments == client_id].copy()

                            # Only process non-empty batches
                            if not client_batch.empty:
                                tasks.append((client_batch, client_id, client_files[client_id], schema_dict))

                        # Process batches in parallel
                        if tasks:
                            results = pool.map(process_batch, tasks)

                            # Update counters
                            for i, result in enumerate(results):
                                if i < len(client_counters):
                                    client_counters[i] += result

                        # Force garbage collection
                        del batch_df
                        del assignments
                        gc.collect()

                    except Exception as e:
                        print(f"Error processing batch {batch_id}: {e}")
                        continue
        else:
            # Sequential processing
            for batch_id in tqdm(range(num_row_groups)):
                try:
                    # Read one row group at a time
                    batch = parquet_file.read_row_group(batch_id)
                    batch_df = batch.to_pandas()
                    batch_size = len(batch_df)

                    # Generate random assignments for this batch
                    assignments = np.random.choice([0, 1, 2], size=batch_size)

                    # Split the batch based on assignments
                    for client_id in range(3):
                        # Filter rows for this client
                        client_batch = batch_df[assignments == client_id]

                        if not client_batch.empty:
                            # Process this batch
                            rows_added = process_batch((client_batch, client_id, client_files[client_id], schema_dict))
                            client_counters[client_id] += rows_added

                    # Force garbage collection
                    del batch_df
                    del assignments
                    gc.collect()

                except Exception as e:
                    print(f"Error processing batch {batch_id}: {e}")
                    continue

        for i in range(3):
            print(f"Client {i + 1}: {client_counters[i]} rows written")

    else:
        raise ValueError("Invalid partitioning method or missing partition column")

    print("Partitioning complete")

    # Final report
    total_written = sum(client_counters) if method != 'row' else total_rows
    print(f"Total rows processed: {total_written} out of {total_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition a large Parquet file for federated learning")
    parser.add_argument("--input", required=True, help="Input Parquet file")
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--method", choices=['row', 'column', 'random'],
                        help="Partitioning method (auto-detected if not specified)")
    parser.add_argument("--column", help="Column name for value-based partitioning")
    parser.add_argument("--batch", type=int, help="Batch size for processing (auto if not specified)")
    parser.add_argument("--processes", type=int, help="Number of processes to use (auto-detected if not specified)")

    args = parser.parse_args()

    if args.method == 'column' and args.column is None:
        parser.error("--column is required when using column-based partitioning")

    # Print some system information
    print(f"Python version: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")

    partition_large_parquet(args.input, args.output, args.method, args.column, args.batch, args.processes)