"""Compute the multimodality
"""
from typing import List
import numpy as np
from numpy import linalg as LA


def multimodality(
    latents_subset1: List[np.ndarray],
    latents_subset2: List[np.ndarray],
) -> float:
    """Compute the multimodality

    Parameters
    ----------
    latents_subset1: List[np.ndarray]
        List of the latent vectors
    latents_subset2: List[np.ndarray]
        List of the latent vectors. It should be aligned with latents_subset1

    Returns
    -------
    float
        multimodality
    """
    if len(latents_subset1) == 0 or len(latents_subset2) == 0:
        return 0

    return np.mean(
        LA.norm(np.array(latents_subset1) - np.array(latents_subset2), axis=1)
    )
