import requests
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
import sys
from matplotlib.animation import FuncAnimation

# Configuration
SERVER_URL = "http://localhost:8080"  # Adjust if needed
POLL_INTERVAL = 5  # Seconds between data refreshes


def fetch_metrics():
    """Fetch training metrics from the central server"""
    try:
        print(f"Fetching metrics from {SERVER_URL}/metrics...")
        response = requests.get(f"{SERVER_URL}/metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"Received data: {len(data)} rounds of metrics")
            return data
        else:
            print(f"Error fetching metrics: {response.status_code}")
            print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"Exception while fetching metrics: {e}")
        return []


def fetch_status():
    """Fetch current server status"""
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching status: {response.status_code}")
            return {"status": "error", "message": f"HTTP error {response.status_code}"}
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return {"status": "error", "message": str(e)}

# Add debugging to see what's being returned
def fetch_status_with_debug():
    """Fetch current server status with debugging"""
    status = fetch_status()
    print(f"DEBUG - Server status response: {status}")
    return status


def update_plot(frame):
    """Update function for the animation"""
    # Get server status first
    status = fetch_status()
    if status:
        plt.suptitle(
            f"Federated Learning Progress - Round {status.get('current_round', '?')}/{status.get('total_rounds', '?')}\n"
            f"Status: {status.get('status', 'Unknown')} - Model: {status.get('model_type', 'Unknown')}")

    # Get metrics data
    metrics = fetch_metrics()

    # Clear previous plots
    plt.clf()

    # Add title back after clearing
    if status:
        plt.suptitle(
            f"Federated Learning Progress - Round {status.get('current_round', '?')}/{status.get('total_rounds', '?')}\n"
            f"Status: {status.get('status', 'Unknown')} - Model: {status.get('model_type', 'Unknown')}")

    if not metrics:
        # No data yet, show a message
        plt.figtext(0.5, 0.5, "Waiting for training data...",
                    ha="center", va="center", fontsize=14,
                    bbox={"boxstyle": "round", "alpha": 0.3})
        return

    # Extract data
    rounds = [m.get('round', i) for i, m in enumerate(metrics)]
    accuracies = [m.get('avg_accuracy', 0) * 100 for m in metrics]  # Convert to percentage

    # Plot accuracy
    plt.subplot(1, 1, 1)
    plt.plot(rounds, accuracies, marker='o', color='blue', linewidth=2)
    plt.title('Average Accuracy Over Rounds')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim([max(0, min(accuracies) - 5), min(100, max(accuracies) + 5)])

    # Add a table of the most recent metrics
    if metrics:
        latest = metrics[-1]
        clients = latest.get('clients', [])

        plt.figtext(0.02, 0.02,
                    f"Latest Round: {latest.get('round')}\n"
                    f"Clients: {', '.join(clients)}\n"
                    f"Accuracy: {latest.get('avg_accuracy', 0) * 100:.2f}%",
                    fontsize=10, bbox={"boxstyle": "round", "alpha": 0.1})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save the current plot
    plt.savefig("fl_progress.png")
    print(f"Updated plot with {len(rounds)} rounds of data")


def main():
    # Set up the plot
    plt.figure(figsize=(10, 6))
    plt.suptitle("Connecting to server...")

    # Check if server is available
    try:
        status = fetch_status()
        if status:
            print(f"Connected to server. Status: {status}")
            plt.suptitle(
                f"Federated Learning Progress - Round {status.get('current_round', '?')}/{status.get('total_rounds', '?')}")
        else:
            print("Could not connect to server or received empty status.")
            plt.figtext(0.5, 0.5, "Could not connect to server.\nCheck if the server is running at " + SERVER_URL,
                        ha="center", va="center", fontsize=12,
                        bbox={"boxstyle": "round", "alpha": 0.3})
            plt.savefig("fl_progress.png")
            plt.show()
            print(f"Please ensure the server is running and accessible at {SERVER_URL}")
            return
    except Exception as e:
        print(f"Error connecting to server: {e}")
        plt.figtext(0.5, 0.5, f"Error connecting to server:\n{str(e)}",
                    ha="center", va="center", fontsize=12,
                    bbox={"boxstyle": "round", "alpha": 0.3})
        plt.savefig("fl_progress.png")
        plt.show()
        return

    # Update once immediately
    update_plot(0)

    # Create animation
    ani = FuncAnimation(plt.gcf(), update_plot, interval=POLL_INTERVAL * 1000)

    # Show plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Allow command-line override of server URL
    if len(sys.argv) > 1:
        SERVER_URL = sys.argv[1]
        print(f"Using server URL: {SERVER_URL}")

    print(f"Visualization starting - connecting to {SERVER_URL}")
    main()