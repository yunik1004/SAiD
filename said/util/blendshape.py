"""Define the utility functions related to the blendshape
"""
import pandas as pd
import torch


def load_blendshape_coeffs(coeffs_path: str) -> torch.FloatTensor:
    """Load the blendshape coefficients file

    Parameters
    ----------
    coeffs_path : str
        Path of the blendshape coefficients file (csv format)

    Returns
    -------
    torch.FloatTensor
        (T_b, B), Blendshape coefficients
    """
    df = pd.read_csv(coeffs_path)
    coeffs = torch.FloatTensor(df.values)
    return coeffs
