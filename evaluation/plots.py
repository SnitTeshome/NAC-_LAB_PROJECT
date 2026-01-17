"""
Plotting utilities for training and evaluation.

Contains functions to visualize and save training/validation loss curves.
"""

import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss(
    train_losses,
    val_losses,
    out_path="evaluation/plots/loss_curve.png",
    title="Training vs Validation Loss"
):
    """
    Plots training and validation loss curves and saves the figure.

    Args:
        train_losses (list[float]): List of training loss values per epoch.
        val_losses (list[float]): List of validation loss values per epoch.
        out_path (str): File path to save the plot image.
        title (str): Title of the plot.

    Returns:
        None
    """
    # Ensure output directory exists
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()