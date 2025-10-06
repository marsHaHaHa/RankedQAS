"""
SoftNDCGLoss
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class SoftNDCGLoss(nn.Module):
    def __init__(self, temperature=1, k=-1, metric_function='L1'):
        super().__init__()
        self.temperature = temperature  # Temperature parameter
        self.k = k  # Truncation parameter (@k)
        self.metric_function = metric_function  # L1 or L2
        self.DEFAULT_EPS = 1e-10

    def DCG(self, scores, labels, k):  # Calculate DCG
        _, indices = torch.sort(scores, descending=True)  # Sort scores to get the ranking indices
        sorted_labels = labels[indices]  # Rearrange labels according to the indices
        discounts = torch.log2(
            torch.arange(len(sorted_labels), dtype=torch.float32) + 2.0)  # Calculate discount factors
        gains = sorted_labels / discounts  # Calculate gains
        if k != -1:
            gains = gains[:k]
        dcg = torch.sum(gains)  # Calculate DCG
        return dcg

    def IDCG(self, labels, k):  # Calculate IDCG
        _, indices = torch.sort(labels, descending=True)  # Sort labels to get the ranking indices
        sorted_labels = labels[indices]  # Rearrange labels according to the indices
        discounts = torch.log2(
            torch.arange(len(sorted_labels), dtype=torch.float32) + 2.0)  # Calculate discount factors
        gains = sorted_labels / discounts  # Calculate gains
        if k != -1:
            gains = gains[:k]
        idcg = torch.sum(gains)  # Calculate IDCG
        return idcg

    def softsort(self, scores: Tensor):  # Calculate the approximate permutation matrix
        temperature = self.temperature  # Temperature parameter

        _, indices = torch.sort(scores, descending=True)

        # Handle the dimension of scores, as the input score shape is (n,) -> (n,1)
        scores = scores.unsqueeze(1)  # Use unsqueeze to add a dimension at index 1

        sorted_scores = scores.sort(descending=True, dim=0)[0]

        # Calculate distances
        if self.metric_function == 'L1':
            ones = torch.ones(len(scores), 1)  # Create a tensor of ones with shape (n,1)
            distance = torch.matmul(sorted_scores, ones.transpose(0, 1)) - torch.matmul(ones,
                                                                                        scores.transpose(0, 1))  # x-y
            distance = distance.abs().neg()
            # print("L1 distance:\n",distance)
        elif self.metric_function == 'L2':
            ones = torch.ones(len(scores), 1)  # Create a tensor of ones with shape (n,1)
            distance = torch.matmul(sorted_scores, ones.transpose(0, 1)) - torch.matmul(ones,
                                                                                        scores.transpose(0, 1))  # x-y
            distance = ((distance) ** 2).neg()
            # print("L2 distance:\n",distance)

        distance = distance / temperature
        # Calculate the approximate permutation matrix
        P_hat = F.softmax(distance, dim=1)
        return P_hat

    def sinkhorn(self, mat, tol=1e-6,
                 max_iter=50):  # Use Sinkhorn algorithm to process the approximate permutation matrix scale(P_hat)
        for _ in range(max_iter):
            mat = mat / mat.sum(dim=0, keepdim=True).clamp(min=self.DEFAULT_EPS)  # Normalize columns
            mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=self.DEFAULT_EPS)  # Normalize rows

            # Check if the sum of each row and column is close to 1 (within the tolerance tol).
            if torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=0) - 1.)) < tol:
                break
        return mat

    def forward(self, y_pred, y_true):
        scores = y_pred
        labels = y_true

        labels = torch.pow(2., labels) - 1.

        # Calculate IDCG
        idcg = self.IDCG(labels, k=self.k)

        # Calculate the approximate permutation matrix
        P_hat = self.softsort(scores)

        # Use Sinkhorn to process the approximate permutation matrix scale(P_hat)
        P_hat = self.sinkhorn(P_hat)

        # Calculate approximate gains: scale(P_hat)g(y)
        approx_gains = torch.matmul(P_hat, labels.float())

        # Calculate approximate DCG
        discounts = torch.log2(torch.arange(len(approx_gains),
                                            dtype=torch.float32) + 2)  # Calculate discount factors, verified to be correct
        approx_gains = approx_gains / discounts  # Calculate approximate gains

        if self.k != -1:  # Truncate, only consider the top k items
            approx_gains = approx_gains[:self.k]
        approx_dcg = torch.sum(approx_gains)  # Calculate approximate DCG

        # Calculate NDCG
        ndcg = approx_dcg / idcg
        # Calculate loss
        loss = 1 - ndcg
        return loss


# Usage test
if __name__ == '__main__':
    # Ground truth labels
    y_true = [1.0, 2.0, 2.0, 4.0, 1.0, 4.0, 3.0]  # Test data from NeuralNDCG paper
    # Predicted scores
    y_pred = [0.5, 0.2, 0.1, 0.4, 1.0, -1.0, 0.63]  # ndcg=0.7668
    # Convert to torch.Tensor
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Instantiate SoftNDCGLoss
    Loss = SoftNDCGLoss()

    print("Test if DCG, IDCG, and NDCG calculation are accurate")
    # Test the DCG function
    DCG = Loss.DCG(y_pred, y_true, k=-1)
    IDCG = Loss.DCG(y_true, y_true, k=-1)
    ndcg = DCG / IDCG
    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("DCG:", DCG)
    print("IDCG:", IDCG)
    print("NDCG:", ndcg)

    # Test softsort
    print("\nTest if softsort is accurate:")
    s = [2, 5, 4]
    print("s:", s)
    s = torch.tensor(s, dtype=torch.float32)
    P_hat = Loss.softsort(s)
    print("P_hat:\n", P_hat)

    # Test sinkhorn
    print("\nTest sinkhorn: The standard is that row and column sums are both 1")
    P_hat_scaled = Loss.sinkhorn(P_hat)
    print("scale(P_hat):\n", P_hat_scaled)
    print("Column sum:", P_hat_scaled.sum(dim=0))
    print("Row sum:", P_hat_scaled.sum(dim=1))

    # Test the forward method of SoftNDCG
    loss = Loss(y_pred, y_true)
    print("\nloss:", loss)