"""
Commonly used ranking metrics calculation:
1. NDCG (Normalized Discounted Cumulative Gain)
2. Spearman's Rank Correlation Coefficient
3. Pearson Correlation Coefficient (mentioned but not implemented)
4. Kendall's Tau Correlation Coefficient
"""
import numpy as np
import torch
from scipy import stats

def NDCG(
    y_pred,
    y_true,
    ats=None,
    gain_function=lambda x: torch.pow(2, x) - 1,
    discounts=None, # Discount factor, to be implemented later TODO
):
    """
    Calculates the Normalized Discounted Cumulative Gain.

    :param y_pred: Predicted scores, shape: (batch_size,)
    :param y_true: True labels, shape: (batch_size,)
    :param ats: Truncation level (at-s). If None, it uses the maximum length y_true.shape[0].
    :param gain_function: Gain function, defaults to 2^x - 1.
    :return: The NDCG score as a scalar value.
    """
    """
    Test data from the TensorFlow official documentation:
    y_pred = torch.tensor([0., 1., 1.])
    y_true = torch.tensor([3., 1., 2.])
    ats = None
    ndcg = NDCG(
        y_pred=y_pred,
        y_true=y_true,
        ats=ats,
    ) # TensorFlow result is 0.6934264, hand-calculated is 0.6806
    """
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)

    dcg = DCG(
        y_pred=y_pred,
        y_true=y_true,
        ats=ats,
        gain_function=gain_function,
    )

    idcg = DCG(
        y_pred=y_true,
        y_true=y_true,
        ats=ats,
        gain_function=gain_function,
    )

    # Add a small epsilon to avoid division by zero
    ndcg = dcg / (idcg + 1e-8)

    return ndcg.item()

def DCG(
        y_pred,
        y_true,
        ats=None,
        gain_function=lambda x: torch.pow(2, x) - 1,
):
    """
    Calculates the Discounted Cumulative Gain.

    :param y_pred: Predicted scores, shape: (batch_size,)
    :param y_true: True labels, shape: (batch_size,)
    :param ats: Truncation level (at-s). If None, it uses the maximum length y_true.shape[0].
    :param gain_function: Gain function, defaults to 2^x - 1.
    :return: The DCG score as a scalar tensor.
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[0]
    if ats is None:
        ats = actual_length
    ats = min(ats, actual_length) # Ensure ats is not greater than the actual length

    # Sort y_pred in descending order and return the indices
    _, indices = y_pred.sort(descending=True)

    # Sort y_true according to the new indices
    y_true_sorted = torch.gather(y_true, dim=0, index=indices)

    # Calculate discount factors, defaults to 1/log2(i+2)
    discounts = (
        torch.tensor(1)
        / torch.log2(
            torch.arange(y_true_sorted.shape[0], dtype=torch.float) + 2.0
        )
    )

    # Calculate gains
    gains = gain_function(y_true_sorted)

    # Calculate discounted gains
    discounted_gains = (gains * discounts)[:ats]

    # Calculate cumulative discounted gains
    dcg = torch.sum(discounted_gains, dim=0)

    return dcg

def spearman_corr(x, y):
    """
    Calculate Spearman's rank correlation coefficient. The order of parameters does not matter.
    :param x: True labels
    :param y: Predicted labels
    :return: Spearman's correlation coefficient
    """
    res = stats.spearmanr(x, y) # The order of parameters does not matter

    return res.statistic

def kendall_corr(x, y):
    """
    Calculate Kendall's tau correlation coefficient. The order of parameters does not matter.
    :param x: True labels
    :param y: Predicted labels
    :return: Kendall's tau correlation coefficient
    """
    res = stats.kendalltau(x, y)

    return res.statistic

def r2_torch(y_true, y_pred):
    """
    R² (coefficient of determination) calculation implemented using PyTorch.

    Args:
        y_true (torch.Tensor): Ground truth values, shape (N,)
        y_pred (torch.Tensor): Predicted values, shape (N,)

    Returns:
        float: The R² score as a scalar value.
    """
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    # Add a small epsilon to avoid division by zero
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2.item()

def get_top_k_energy(y_pred, y_true, k):
    """
    Finds the best (minimum) energy among the top-k predictions.
    This assumes that higher y_true values correspond to lower energy.
    :param y_pred: Predicted scores.
    :param y_true: True values (e.g., negated energy values, where higher is better).
    :param k: The number of top elements to consider.
    :return: The best energy value (maximum of the y_true values) among the top-k predictions.
    """
    rank = np.argsort(np.array(y_pred))
    picked_index = rank[::-1] # Indices of predictions from highest to lowest score

    labels_top_k = []
    for i in range(0, k):
        labels_top_k.append(y_true[picked_index[i]])

    labels_top_k = np.array(labels_top_k)
    # The max of the negated energy values corresponds to the minimum actual energy
    energy_min = np.max(labels_top_k)

    return energy_min