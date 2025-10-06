"""
Plotting loss curves.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

# Assuming ensure_dir is a utility function you have.
from utils.common import ensure_dir


def basic_figure(
        x, y,
        title,
        color='blue',
        save_path=None,
        show=False,
        format='.png',
):
    """
    Creates a basic plot.

    Args:
        x (iterable): Data for the x-axis.
        y (iterable): Data for the y-axis.
        title (str): Title of the plot.
        color (str): Color of the plot line.
        save_path (str, optional): Directory to save the plot. Defaults to None.
        show (bool): Whether to display the plot. Defaults to False.
        format (str): Image format. Defaults to '.png'.

    Returns:
        tuple: The figure and axes objects.
    """
    fig, ax = plt.subplots()  # Create a figure and a set of subplots
    # fig, axs = plt.subplots(2, 2) # Example for a 2x2 grid of subplots
    ax.plot(x, y, color=color, label=r'$y=x^2$')  # Example label
    ax.set(
        xlabel='x',
        ylabel='y',
        title=title,
    )
    ax.legend()
    if save_path is not None:
        # It's safer to join paths
        full_save_path = os.path.join(save_path, f"{title}{format}")
        plt.savefig(full_save_path)
    if show:
        plt.show()
    plt.close(fig)  # Close the figure to free memory
    return fig, ax


def loss_curve(
        loss,
        title,
        label='Train Loss',
        color='blue',
        save_path=None,
        save_name='train_loss',
        show=False,
        format='.png',
):
    """
    Plots a single loss curve.

    Args:
        loss (list): Loss values, e.g., training loss or validation loss.
        title (str): The title of the plot, e.g., 'Train Loss' or 'Valid Loss'.
        label (str): The label for the curve.
        color (str): The color of the curve. Defaults to blue for training, red for validation.
        save_path (str, optional): The directory to save the plot.
        save_name (str): The name of the file to save.
        show (bool): Whether to display the plot. Defaults to False for batch processing.
        format (str): Image format. Can be changed to '.pdf' or '.svg' if needed.

    Returns:
        tuple: The figure and axes objects, useful for integration with tools like TensorBoard.
    """
    # Process loss data
    x = range(1, len(loss) + 1)
    y = loss

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, label=label)
    ax.set(
        xlabel='Iterations',
        ylabel='Loss',
        title=title,
    )
    ax.legend()

    if save_path is not None:
        ensure_dir(save_path)
        full_save_path = os.path.join(save_path, f"{save_name}{format}")
        plt.savefig(full_save_path)

    if show:
        plt.show()

    plt.close(fig)
    return fig, ax


def plot_train_and_val_loss(
        train_loss,
        valid_loss,
        title='Loss Curve',
        xlabel='Epochs',
        ylabel='Loss',
        save_path=None,
        save_name='loss',
        show=False,
        format='.png',
):
    """
    Plots both training and validation loss curves on the same graph.

    Args:
        train_loss (list): A list of training loss values.
        valid_loss (list): A list of validation loss values.
        title (str): The chart title.
        xlabel (str): The label for the X-axis.
        ylabel (str): The label for the Y-axis.
        save_path (str, optional): The path to save the figure.
        save_name (str): The filename for the saved figure (without extension).
        show (bool): Whether to display the plot.
        format (str): The image format.

    Returns:
        tuple: The figure and axes objects.
    """
    x_train = range(1, len(train_loss) + 1)
    y_train = train_loss
    x_valid = range(1, len(valid_loss) + 1) if valid_loss else []
    y_valid = valid_loss

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_train, y_train, 'b-o', markersize=4, label='Training Loss')
    if y_valid:
        ax.plot(x_valid, y_valid, 'r-s', markersize=4, linestyle='--', label='Validation Loss')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)

    if save_path:
        ensure_dir(save_path)
        # Using os.path.join is safer for constructing file paths.
        full_save_path = os.path.join(save_path, f"{save_name}{format}")
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Loss curve plot saved to: {full_save_path}")

    if show:
        plt.show()

    plt.close(fig)  # Close the figure to free up memory
    return fig, ax