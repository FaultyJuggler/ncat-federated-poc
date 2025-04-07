import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_classification
from sklearn.model_selection import train_test_split


def generate_mnist_parquet():
    """Generate Parquet files from MNIST dataset"""
    print("Downloading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target

    # Create a dataframe
    cols = [f'pixel_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y

    # Split data for three clients
    total_samples = len(df)
    client1 = df.iloc[:total_samples // 3]
    client2 = df.iloc[total_samples // 3:2 * total_samples // 3]
    client3 = df.iloc[2 * total_samples // 3:]

    # Create data directory
    os.makedirs("data/client1", exist_ok=True)
    os.makedirs("data/client2", exist_ok=True)
    os.makedirs("data/client3", exist_ok=True)

    # Save to Parquet files
    client1.to_parquet("data/client1/dataset1.parquet", index=False)
    client2.to_parquet("data/client2/dataset2.parquet", index=False)
    client3.to_parquet("data/client3/dataset3.parquet", index=False)

    print(f"Generated Parquet files with MNIST data:")
    print(f"Client 1: {len(client1)} samples")
    print(f"Client 2: {len(client2)} samples")
    print(f"Client 3: {len(client3)} samples")


def generate_synthetic_parquet():
    """Generate Parquet files with synthetic data"""
    print("Generating synthetic classification data...")

    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=15000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=5,
        random_state=42
    )

    # Create a dataframe with the data
    cols = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y

    # Add some categorical features
    df['category_1'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    df['category_2'] = np.random.choice(['X', 'Y', 'Z', 'W'], size=len(df))

    # Split data for three clients
    total_samples = len(df)
    client1 = df.iloc[:total_samples // 3]
    client2 = df.iloc[total_samples // 3:2 * total_samples // 3]
    client3 = df.iloc[2 * total_samples // 3:]

    # Create data directory
    os.makedirs("data/client1", exist_ok=True)
    os.makedirs("data/client2", exist_ok=True)
    os.makedirs("data/client3", exist_ok=True)

    # Save to Parquet files
    client1.to_parquet("data/client1/synthetic1.parquet", index=False)
    client2.to_parquet("data/client2/synthetic2.parquet", index=False)
    client3.to_parquet("data/client3/synthetic3.parquet", index=False)

    print(f"Generated Parquet files with synthetic data:")
    print(f"Client 1: {len(client1)} samples")
    print(f"Client 2: {len(client2)} samples")
    print(f"Client 3: {len(client3)} samples")


if __name__ == "__main__":
    print("Generating sample Parquet files for federated learning...")

    choice = input("Choose dataset type (1 for MNIST, 2 for Synthetic): ")

    if choice == '1':
        generate_mnist_parquet()
    elif choice == '2':
        generate_synthetic_parquet()
    else:
        print("Invalid choice. Generating both datasets.")
        generate_mnist_parquet()
        generate_synthetic_parquet()

    print("Done!")