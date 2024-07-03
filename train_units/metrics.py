"""
author: Ned
"""

import torch


def rmse(y_pred, y_true):
    """RMSE"""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))


def mae(y_pred, y_true):
    """MAE"""
    return torch.mean(torch.abs(y_true - y_pred))


def r2_score(y_pred, y_true):
    """R^2"""
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
