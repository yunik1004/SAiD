"""Define the utility functions related to the audio
"""
from dataclasses import dataclass
import math
from typing import List
import librosa
import numpy as np
import torch
import torchaudio


@dataclass
class FittedWaveform:
    """Fitted waveform using the window"""

    waveform: torch.FloatTensor
    window_size: int


def load_audio(audio_path: str, sampling_rate: int) -> torch.FloatTensor:
    """Load the audio file

    Parameters
    ----------
    audio_path : str
        Path of the audio file
    sampling_rate : int
        Sampling rate of the output audio wave

    Returns
    -------
    torch.FloatTensor
        (T_a), Mono waveform
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != sampling_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
    waveform_mono = torch.mean(waveform, dim=0)
    return waveform_mono


def fit_audio_unet(
    waveform: torch.FloatTensor, sampling_rate: int, fps: int, divisor_unet: int
) -> FittedWaveform:
    """Fit the intput audio waveform into UNet1D

    Parameters
    ----------
    waveform : torch.FloatTensor
        (T_a), Mono waveform
    sampling_rate : int
        Sampling rate of the audio model
    fps : int
        The number of frames per second
    divisor_unet : int
        Length of the blendshape coefficients sequence should be divided by this number

    Returns
    -------
    FittedWaveform
        Fitted waveform with the window
    """
    gcd = math.gcd(sampling_rate, fps)
    divisor_waveform = sampling_rate // gcd * divisor_unet

    waveform_len = waveform.shape[0]
    window_len = int(waveform_len / sampling_rate * fps)
    waveform_len_fit = math.ceil(waveform_len / divisor_waveform) * divisor_waveform

    if waveform_len_fit > waveform_len:
        tmp = torch.zeros(waveform_len_fit)
        tmp[:waveform_len] = waveform[:]
        waveform = tmp

    return FittedWaveform(waveform=waveform, window_size=window_len)


def compute_audio_beat_time(waveform: np.ndarray, sampling_rate: int) -> List[float]:
    """Compute the audio beat time

    Parameters
    ----------
    waveform : np.ndarray
        (T_a), Mono waveform
    sampling_rate : int
        Sampling rate of the audio

    Returns
    -------
    List[float]
        Audio beat time (secs)
    """
    audio_beat_time = librosa.onset.onset_detect(
        y=waveform, sr=sampling_rate, units="time"
    )
    return audio_beat_time
