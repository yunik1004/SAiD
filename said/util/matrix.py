"""Define the utility functions related to the matrix
"""
import numpy as np
import torch


def singleton(class_):
    instances = {}

    def get_instance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return get_instance


def band_matrix(size: int, bandwidth: int) -> np.ndarray:
    return np.abs(np.add.outer(np.arange(size), -np.arange(size))) < bandwidth + 1


@singleton
class BandMatrices:
    def __init__(self, size: int = 1000) -> None:
        self.matrices = torch.from_numpy(
            np.stack([band_matrix(size, w) for w in range(size)])
        )  # (size, size, size)

    def get_matrices(self, size: int, widths) -> torch.BoolTensor:
        return self.matrices[widths.cpu(), :size, :size]
