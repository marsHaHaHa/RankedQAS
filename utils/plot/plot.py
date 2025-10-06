import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
# Assuming common.py is in a directory like 'utils/'
from ..common import ensure_dir


# --- Core Plotting Functions ---

def plot_metric_curves(
        train_metric_values,
        valid_metric_values,
        metric_name,
        title=None,
        save_path=None,
        save_name='metric_curve.png',
        show_markers=False,
        show=False,
):
    """
    Plots and saves metric curves for training and validation sets with a consistent and aesthetically pleasing style.

    Args:
        train_metric_values (list): A list of metric values for the training set.
        valid_metric_values (list): A list of metric values for the validation set.
        metric_name (str): The name of the metric, used for the Y-axis label and legend (e.g., 'Loss', 'Spearman', 'NDCG@10').
        title (str, optional): The chart title. If None, a title will be generated automatically.
        save_path (str, optional): The directory to save the image.
        save_name (str): The full filename for the saved image.
        show_markers (bool): Whether to show data point markers on the curves.
        show (bool): Whether to display the plot at runtime.
    """
    epochs = range(1, len(train_metric_values) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    if title is None:
        title = f'{metric_name} Trend'

    train_style = {'color': 'dodgerblue', 'linewidth': 2.5, 'label': f'Training {metric_name}'}
    valid_style = {'color': 'orangered', 'linestyle': '--', 'linewidth': 2.5, 'label': f'Validation {metric_name}'}
    if show_markers:
        train_style.update({'marker': 'o', 'markersize': 4})
        valid_style.update({'marker': 's', 'markersize': 4})

    ax.plot(epochs, train_metric_values, **train_style)
    ax.plot(epochs, valid_metric_values, **valid_style)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)

    all_values = train_metric_values + valid_metric_values
    if all_values:
        min_val, max_val = min(all_values), max(all_values)
        margin = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
        ax.set_ylim(min_val - margin, max_val + margin)

    ax.legend(fontsize=12, loc='best')
    ax.grid(True, linestyle=':', alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if save_path:
        ensure_dir(save_path)
        full_save_path = os.path.join(save_path, save_name)
        plt.savefig(full_save_path, bbox_inches='tight', dpi=300)
        print(f"'{title}' saved to: {full_save_path}")

    if show:
        plt.show()

    plt.close(fig)


def scatter_plot(
        pred,
        target,
        metrics: dict,  # Dictionary to pass in metrics
        title='Predicted vs. True Values',
        save_path=None,
        save_name='scatter',
        show=False,
        format='.png',
):
    """
    Plots a classic scatter plot of predicted vs. true values, featuring a full border,
    a dense grid, and metric information in the bottom-right corner.

    Args:
        pred (list or np.array): The model's predicted values.
        target (list or np.array): The true label values.
        metrics (dict): A dictionary containing additional metrics, e.g.,
                        {'spearman': 0.8, 'ndcg@10': 0.9, 'ndcg@20': 0.95}.
        title (str): The chart title.
        save_path (str, optional): The directory to save the image.
        save_name (str): The filename for the saved image (without extension).
        show (bool): Whether to display the plot at runtime.
        format (str): The image format.

    Returns:
        tuple: (fig, ax) The Matplotlib figure and axes objects.
    """
    print(f"Plotting scatter plot with {len(pred)} points.")

    x = np.array(pred).flatten()
    y = np.array(target).flatten()

    # Create the figure
    fig, ax = plt.subplots(figsize=(9, 8))

    # Plot scatter using seaborn
    sns.scatterplot(x=x, y=y, color='royalblue', alpha=0.6, ax=ax, edgecolor='none', s=40)

    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('True Values', fontsize=12)

    # [Core Change] Build and display the metric text box in the bottom-right corner
    spearman_val = metrics.get('spearman', 'N/A')
    ndcg10_val = metrics.get('ndcg@10', 'N/A')
    ndcg20_val = metrics.get('ndcg@20', 'N/A')

    text_lines = []
    if spearman_val != 'N/A':
        text_lines.append(f'Spearman ρ = {spearman_val:.3f}')
    if ndcg10_val != 'N/A':
        text_lines.append(f'NDCG@10 = {ndcg10_val:.3f}')
    if ndcg20_val != 'N/A':
        text_lines.append(f'NDCG@20 = {ndcg20_val:.3f}')

    info_text = '\n'.join(text_lines)

    if info_text:
        # Place the text box in the bottom-right corner
        ax.text(0.95, 0.05, info_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='whitesmoke', alpha=0.8, edgecolor='gray'))

    # [Core Change] Add a denser grid
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)

    # By default, all spines are visible, so we don't need to explicitly set them to True.
    # We just need to ensure they are not set to False.

    plt.tight_layout()

    # Save the figure to the specified path
    if save_path is not None:
        ensure_dir(save_path)
        full_save_path = os.path.join(save_path, f"{save_name}{format}")
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"'{title}' saved to: {full_save_path}")

    if show:
        plt.show()

    plt.close(fig)
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

    # Add grid lines, moderately dense
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


# --- Specialized Interface Functions ---

def plot_loss_curves(train_loss, valid_loss, title, save_path, save_name, show=False):
    """Specialized interface for plotting loss curves."""
    plot_metric_curves(
        train_metric_values=train_loss,
        valid_metric_values=valid_loss,
        metric_name='Loss',
        title=title,
        save_path=save_path,
        save_name=save_name,
        show=show
    )


def plot_spearman_curves(train_spearman, valid_spearman, title, save_path, save_name, show=False):
    """Specialized interface for plotting Spearman correlation coefficient curves."""
    plot_metric_curves(
        train_metric_values=train_spearman,
        valid_metric_values=valid_spearman,
        metric_name='Spearman Correlation',
        title=title,
        save_path=save_path,
        save_name=save_name,
        show=show
    )


def plot_ndcg_curves(train_ndcg, valid_ndcg, k, save_path, save_name, show=False):
    """Specialized interface for plotting NDCG@k curves."""
    metric_name = f'NDCG@{k}'
    title = f'{metric_name} Trend'
    plot_metric_curves(
        train_metric_values=train_ndcg,
        valid_metric_values=valid_ndcg,
        metric_name=metric_name,
        title=title,
        save_path=save_path,
        save_name=save_name,
        show=show
    )


# R2 Curve
def plot_r2_curves(train_r2, valid_r2, title, save_path, save_name, show=False):
    """Specialized interface for plotting R2 score curves."""
    plot_metric_curves(
        train_metric_values=train_r2,
        valid_metric_values=valid_r2,
        metric_name='R2 Score',
        title=title,
        save_path=save_path,
        save_name=save_name,
        show=show
    )


# MAE Curve
def plot_mae_curves(train_mae, valid_mae, title, save_path, save_name, show=False):
    """Specialized interface for plotting MAE curves."""
    plot_metric_curves(
        train_metric_values=train_mae,
        valid_metric_values=valid_mae,
        metric_name='MAE',
        title=title,
        save_path=save_path,
        save_name=save_name,
        show=show
    )