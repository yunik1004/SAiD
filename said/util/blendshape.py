"""Define the utility functions related to the blendshape
"""
import pickle
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image
import torch


def load_blendshape_deltas(
    blendshape_deltas_path: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the blendshape deltas

    Parameters
    ----------
    blendshape_deltas_path : str
        Path of the blendshape deltas file

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        {
            "<Person ID>": {
                "<Blendshape name>": np.ndarray, (|V|, 3)
            }
        }
    """
    with open(blendshape_deltas_path, "rb") as f:
        blendshape_deltas = pickle.load(f)

    return blendshape_deltas


def load_blendshape_coeffs(coeffs_path: str) -> torch.FloatTensor:
    """Load the blendshape coefficients file

    Parameters
    ----------
    coeffs_path : str
        Path of the blendshape coefficients file (csv format)

    Returns
    -------
    torch.FloatTensor
        (T_b, num_classes), Blendshape coefficients
    """
    df = pd.read_csv(coeffs_path)
    coeffs = torch.FloatTensor(df.values)
    return coeffs


def save_blendshape_coeffs(
    coeffs: np.ndarray, classes: List[str], output_path: str
) -> None:
    """Save the blendshape coefficients into the file

    Parameters
    ----------
    coeffs : np.ndarray
        (T_b, num_classes), Blendshape coefficients
    classes : List[str]
        List of the class names of the coefficients
    output_path : str
        Path of the output file
    """
    pout = pd.DataFrame(coeffs, columns=classes)
    pout.to_csv(output_path, index=False)


def save_blendshape_coeffs_image(coeffs: np.ndarray, output_path: str) -> None:
    """Save the blendshape coefficients into the image file

    Parameters
    ----------
    coeffs : np.ndarray
        (T_b, num_classes), Blendshape coefficients
    output_path : str
        Path of the output file
    """
    orig = (255 * coeffs.transpose()).round()
    img = Image.fromarray(orig).convert("L")
    img.save(output_path)
