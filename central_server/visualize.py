import requests
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
from matplotlib.animation import FuncAnimation

SERVER_URL = "http://localhost:8080"  # Adjust if needed


def fetch_metrics():
    """Fetch training metrics from the central server"""
    try:
        response = requests.get(f"{SERVER_URL}/metrics")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching metrics: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception while fetching metrics: {e}")
        return []


def update_plot(frame):
    """Update function for the animation"""
    metrics = fetch_metrics()
    if not metrics:
        return

    # Clear previous plots
    plt.clf()

    # Extract data
    rounds = [m['round'] for m in metrics]
    accuracies = [m['avg_accuracy'] for m in metrics]

    # Plot accuracy
    plt.plot(rounds, accuracies, marker='o', color='green')
    plt.title('Average Accuracy Over Rounds (RandomForest)')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.ylim([0, 100])

    plt.tight_layout()

    # Save the current plot
    plt.savefig("rf_progress.png")


if __name__ == "__main__":
    # Set up the plot
    plt.figure(figsize=(10, 6))

    # Create animation
    ani = FuncAnimation(plt.gcf(), update_plot, interval=5000)  # Update every 5 seconds

    # Show plot
    plt.tight_layout()
    plt.show()