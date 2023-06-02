"""Compute the beat consistency score
"""
from typing import List
import numpy as np
from scipy.signal import find_peaks
from said.util.audio import compute_audio_beat_time


def beat_consistency_score(
    list_waveform: List[np.ndarray],
    list_blendshape_coeffs: List[np.ndarray],
    sampling_rate: int,
    fps: int,
    threshold: float,
    sigma: float = 0.1,
) -> float:
    """Compute the beat consistency score

    Parameters
    ----------
    list_waveform: List[np.ndarray]
        List of the waveforms, shape: (audio_sequence_len,)
    list_blendshape_coeffs: List[np.ndarray]
        List of the corresponding blendshape coefficients, shape: (blendshape_seq_len, num_blendshapes)
    sampling_rate: int
        Sampling rate of the waveform
    fps: int
        FPS of the blendshape coefficients sequence
    threshold: float
        Threshold for finding peaks
    sigma: float
        Parameter to normalize sequences, by default 0.1

    Returns
    -------
    float
        Beat consistency score
    """
    # Find audio beats
    list_audio_beats = [
        compute_audio_beat_time(waveform, sampling_rate) for waveform in list_waveform
    ]

    # Find kinematic beats
    list_coeffs_diff = [
        np.abs(coeffs[1:] - coeffs[:-1]) for coeffs in list_blendshape_coeffs
    ]
    mac = np.mean(
        [coeffs_diff.mean(0) for coeffs_diff in list_coeffs_diff], axis=0, keepdims=True
    )
    list_coeffs_change_rate = [
        np.mean(coeffs_diff / mac, axis=1) for coeffs_diff in list_coeffs_diff
    ]

    list_kinematic_beats = []
    for coeffs_change_rate in list_coeffs_change_rate:
        optima_indices, optima_heights = find_peaks(-coeffs_change_rate, threshold=0)
        mask = np.logical_or(
            optima_heights["left_thresholds"] > threshold,
            optima_heights["right_thresholds"] > threshold,
        )
        list_kinematic_beats.append(optima_indices[mask] / fps)

    # Compute beat consistency score
    list_bc = []
    for audio_beats, kinematic_beats in zip(list_audio_beats, list_kinematic_beats):
        bc_single = 0
        if len(kinematic_beats) > 0:
            bc_single = np.mean(
                np.exp(
                    -np.power(
                        audio_beats[:, np.newaxis] - kinematic_beats[np.newaxis, :], 2
                    ).min(axis=1)
                    / (2 * sigma**2)
                )
            )
        list_bc.append(bc_single)

    return np.mean(list_bc)
