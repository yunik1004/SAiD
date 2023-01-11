"""Define the dataset which are used in the scripts
"""

import os
import random
from typing import Any, Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset
from said.util.audio import load_audio
from said.util.blendshape import load_blendshape_coeffs


class VOCARKitTrainDataset(Dataset):
    """Train dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        sampling_rate: int,
        window_size: int = 120,
        fps: int = 60,
    ):
        """Constructor of the class

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        sampling_rate : int
            Sampling rate of the audio
        window_size : int, optional
            Window size of the blendshape coefficients, by default 120
        fps : int, optional
            fps of the blendshape coefficients, by default 60
        """
        self.audio_dir = audio_dir
        self.blendshape_coeffs_dir = blendshape_coeffs_dir
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.fps = fps

        self.audio_paths = [
            os.path.join(self.audio_dir, f) for f in sorted(os.listdir(self.audio_dir))
        ]
        self.blendshape_coeffs_paths = [
            os.path.join(self.blendshape_coeffs_dir, f)
            for f in sorted(os.listdir(self.blendshape_coeffs_dir))
        ]

        self.length = len(self.audio_paths)

    def __len__(self) -> int:
        """Return the size of the dataset

        Returns
        -------
        int
            Size of the dataset
        """
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        """Return the item of the given index

        Parameters
        ----------
        index : int
            Index of the item

        Returns
        -------
        Dict[str, torch.FloatTensor]
            {
                "waveform": (audio_seq_len,),
                "blendshape_coeffs": (num_blendshapes, blendshape_seq_len),
            }
        """

        waveform = load_audio(self.audio_paths[index], self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(
            self.blendshape_coeffs_paths[index]
        ).transpose(0, 1)

        num_blendshape = blendshape_coeffs.shape[0]
        blendshape_len = blendshape_coeffs.shape[1]
        waveform_len = waveform.shape[0]
        waveform_patch_len = (self.sampling_rate * self.window_size) // self.fps

        waveform_patch = None
        blendshape_coeffs_patch = None
        if blendshape_len >= self.window_size:
            idx = random.randint(0, blendshape_len - self.window_size)
            blendshape_coeffs_patch = blendshape_coeffs[:, idx : idx + self.window_size]

            waveform_patch_idx = (self.sampling_rate * idx) // self.fps
            waveform_patch = waveform[
                waveform_patch_idx : waveform_patch_idx + waveform_patch_len
            ]
        else:
            blendshape_coeffs_patch = torch.zeros((num_blendshape, self.window_size))
            blendshape_coeffs_patch[:, :blendshape_len] = blendshape_coeffs[:, :]

            waveform_patch = torch.zeros(waveform_patch_len)
            waveform_patch[:waveform_len] = waveform[:]

        out = {
            "waveform": waveform_patch,
            "blendshape_coeffs": blendshape_coeffs_patch,
        }

        return out

    @staticmethod
    def collate_fn(examples: List[Dict[str, torch.FloatTensor]]) -> Dict[str, Any]:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[Dict[str, torch.FloatTensor]]
            _description_

        Returns
        -------
        Dict[str, Any]
            {
                "waveform": List[np.ndarray], each: (audio_seq_len,)
                "blendshape_coeffs": torch.FloatTensor, (Batch, num_blendshapes, blendshape_seq_len)
            }
        """
        examples_dict = {k: [dic[k] for dic in examples] for k in examples[0]}

        waveforms = [np.array(wave) for wave in examples_dict["waveform"]]
        blendshape_coeffs = torch.stack(examples_dict["blendshape_coeffs"])

        out = {
            "waveform": waveforms,  # List[np.ndarray]
            "blendshape_coeffs": blendshape_coeffs,  # torch.FloatTensor
        }

        return out
