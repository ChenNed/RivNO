import numpy
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

water_depths = torch.Tensor([92.35, 92.45, 92.55, 92.65, 92.75, 92.85, 92.95, 93.05, 93.15, 93.25, 93.35,
                             93.45])


def minmax(data):
    # Min-Max normalization
    min_val = torch.min(data)
    max_val = torch.max(data)
    data_normalized = (data - min_val) / (max_val - min_val)
    return data_normalized


def zscore(data):
    # Z-Score normalization
    mean = torch.mean(data)
    std = torch.std(data)
    data_normalized = (data - mean) / std
    return data_normalized


def minmaxNorWD(data):
    min_val = 92.368
    max_val = 93.487
    data_normalized = (data - min_val) / (max_val - min_val)
    return data_normalized


def zNorWD(data):
    mean = 92.55435
    std = 0.1843659
    data_normalized = (data - mean) / std
    return data_normalized
