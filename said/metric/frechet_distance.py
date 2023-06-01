"""Compute the frechet distance
"""
from dataclasses import dataclass
from typing import List
import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance


@dataclass
class Statistic:
    """Dataclass for the statistic"""

    mean: np.ndarray
    cov: np.ndarray


def get_statistic(data: List[np.ndarray]) -> Statistic:
    """Compute the statistic of the data

    Parameters
    ----------
    data: List[np.ndarray]
        (z_dim,), List of the numpy 1-d arrays

    Returns
    -------
    Statistics
        mean, covariance of the data
    """
    mean = np.mean(data, axis=0)
    covariance = np.cov(data, rowvar=False)

    return Statistic(
        mean=mean,
        cov=covariance,
    )


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """Compute the Frechet distance between two multivariate Gaussians
    X1 ~ N(mu1, sigma1) and X2 ~ N(mu2, sigma2)

    Parameters
    ----------
    mu1: np.ndarray
        Mean of features 1
    sigma1: np.ndarray
        Covariance of features 1
    mu2: np.ndarray
        Mean of features 2
    sigma2: np.ndarray
        Covariance of features 2

    Returns
    -------
    float
        Frechet distance
    """
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
