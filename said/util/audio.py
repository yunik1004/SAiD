"""Define the utility functions related to the audio
"""
import torch
import torchaudio


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
