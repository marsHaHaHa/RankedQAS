"""
Plotting curves, scatter plots, etc., for various metrics.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.common import ensure_dir

def spearman_curve(
    spearman,
    title,
    label='Train Spearman',
    color='blue',
    save_path=None,
    save_name='train_spearman',
    show=False,
    format='.png',
):
    """
    Plots the Spearman correlation coefficient curve for a single set of data.

    Args:
        spearman (list): List of Spearman correlation values.
        title (str): The title of the plot.
        label (str): The label for the data series.
        color (str): The color of the plot line.
        save_path (str, optional): The directory to save the plot.
        save_name (str): The filename for the saved plot.
        show (bool): Whether to display the plot at runtime.
        format (str): The file format for the saved plot.

    Returns:
        tuple: The figure and axes objects from Matplotlib.
    """
    # Prepare data
    x = range(1, len(spearman) + 1)
    y = spearman

    # Create plot
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, label=label)
    ax.set(
        xlabel='Epoch',
        ylabel='Spearman Correlation',
        title=title,
    )
    ax.legend()

    if save_path is not None:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, f"{save_name}{format}"))

    if show:
        plt.show()

    plt.close(fig)
    return fig, ax


def plot_train_and_val_spearman(
        train_spearman,
        valid_spearman,
        title='Spearman Correlation Coefficient Curve',
        save_path=None,
        save_name='spearman_curve.png',
        show=False,
):
    """
    Plots and saves the Spearman correlation coefficient curves for training and validation sets.

    Args:
        train_spearman (list): List of Spearman values for the training set.
        valid_spearman (list): List of Spearman values for the validation set.
        title (str): The title of the chart.
        save_path (str): The directory to save the image.
        save_name (str): The full filename for the saved image (e.g., 'spearman.png').
        show (bool): Whether to display the plot at runtime.

    Returns:
        tuple: (fig, ax) The Matplotlib figure and axes objects.
    """
    # Prepare data
    epochs = range(1, len(train_spearman) + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curves
    ax.plot(epochs, train_spearman, 'o-', color='dodgerblue', markersize=4, linewidth=2, label='Training Spearman')
    ax.plot(epochs, valid_spearman, 's--', color='orangered', markersize=4, linewidth=2, label='Validation Spearman')

    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Spearman Correlation Coefficient', fontsize=12)

    # Set Y-axis range for better visualization
    # Can be adjusted based on your data range, e.g., -1.1 to 1.1
    min_val = min(min(train_spearman), min(valid_spearman))
    max_val = max(max(train_spearman), max(valid_spearman))
    ax.set_ylim(max(-1.05, min_val - 0.1), min(1.05, max_val + 0.1))

    # Add legend and grid
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Optimize borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the figure
    if save_path:
        ensure_dir(save_path)
        full_save_path = os.path.join(save_path, save_name)
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        print(f"'{title}' saved to: {full_save_path}")

    if show:
        plt.show()

    plt.close(fig)  # Close the figure to free up memory
    return fig, ax


def scatter_plot_basic(
    pred,
    target,
    title='Scatter Plot',
    save_path=None,
    save_name='scatter',
    show=False,
    format='.png',
):
    # Plot scatter plot
    print(f"Plotting scatter plot with {len(pred)} points.")

    x = pred
    y = target
    fig, axes = plt.subplots(1, 1, figsize=(14, 14))
    sns.scatterplot(x=x, y=y, color='steelblue', alpha=0.6, ax=axes)
    axes.set_title(title)
    axes.set_xlabel('Prediction')
    axes.set_ylabel('Label')

    # Add grid lines, moderately spaced
    axes.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    x_ticks = np.linspace(min(x), max(x), 6)
    y_ticks = np.linspace(min(y), max(y), 6)
    axes.set_xticks(x_ticks)
    axes.set_yticks(y_ticks)

    plt.tight_layout()

    # Save the figure to the specified path
    if save_path is not None:
        ensure_dir(save_path)
        plt.savefig(os.path.join(save_path, f"{save_name}{format}"), dpi=600, bbox_inches='tight')

    if show:
        plt.show()

    plt.close(fig)
    return fig, axes


def plot_train_and_val_NDCG(
        train_NDCG,
        valid_NDCG,
        title='NDCG Score Curve',
        save_path=None,
        save_name='ndcg_curve.png',
        show=False,
):
    """
    Plots and saves the NDCG score curves for training and validation sets.

    Args:
        train_NDCG (list): List of NDCG values for the training set.
        valid_NDCG (list): List of NDCG values for the validation set.
        title (str): The title of the chart.
        save_path (str): The directory to save the image.
        save_name (str): The full filename for the saved image (e.g., 'ndcg.png').
        show (bool): Whether to display the plot at runtime.

    Returns:
        tuple: (fig, ax) The Matplotlib figure and axes objects.
    """
    # Prepare data
    epochs = range(1, len(train_NDCG) + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot curves
    ax.plot(epochs, train_NDCG, 'o-', color='teal', markersize=4, linewidth=2, label='Training NDCG')
    ax.plot(epochs, valid_NDCG, 's--', color='darkorange', markersize=4, linewidth=2, label='Validation NDCG')

    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('NDCG Score', fontsize=12)

    # Set Y-axis range, NDCG is typically between [0, 1]
    min_val = min(min(train_NDCG), min(valid_NDCG))
    max_val = max(max(train_NDCG), max(valid_NDCG))
    ax.set_ylim(max(-0.05, min_val - 0.1), min(1.05, max_val + 0.1))

    # Add legend and grid
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Optimize borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the figure
    if save_path:
        ensure_dir(save_path)
        full_save_path = os.path.join(save_path, save_name)
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        print(f"'{title}' saved to: {full_save_path}")

    if show:
        plt.show()

    plt.close(fig)  # Close the figure to free up memory
    return fig, ax


def plot_pretrain_metric_curve(train_metric, valid_metric, metric_name, title, save_path, save_name):
    """
    Plots and saves training and validation curves for a single metric during pre-training.

    Args:
        train_metric (list): List of metric values for each epoch on the training set.
        valid_metric (list): List of metric values for each epoch on the validation set.
        metric_name (str): The name of the metric (e.g., 'Spearman Correlation', 'R²').
        title (str): The title of the chart.
        save_path (str): The directory to save the image.
        save_name (str): The filename for the saved image (including extension).
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_metric) + 1)

    # Plot curves
    plt.plot(epochs, train_metric, 'b-o', markersize=4, label=f'Training {metric_name}')
    plt.plot(epochs, valid_metric, 'r-s', linestyle='--', markersize=4, label=f'Validation {metric_name}')

    # Set title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)

    # Add legend and grid
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)

    # Save the figure
    if save_path:
        ensure_dir(save_path)
        full_save_path = os.path.join(save_path, save_name)
        # bbox_inches='tight' ensures labels are not clipped
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        print(f"'{title}' curve plot saved to: {full_save_path}")

    # Close the plot to prevent display in environments like Jupyter and to free memory
    plt.close()