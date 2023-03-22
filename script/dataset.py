"""Define the dataset which are used in the scripts
"""
from abc import abstractmethod, ABC
import glob
import os
import random
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import trimesh
from said.util.audio import load_audio
from said.util.blendshape import load_blendshape_coeffs

VOCARKIT_CLASSES = [
    "jawForward",
    "jawLeft",
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "noseSneerLeft",
    "noseSneerRight",
]

VOCARKIT_CLASSES_MIRROR_PAIR = [
    ("jawLeft", "jawRight"),
    ("mouthLeft", "mouthRight"),
    ("mouthSmileLeft", "mouthSmileRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthPressLeft", "mouthPressRight"),
    ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"),
    ("cheekSquintLeft", "cheekSquintRight"),
    ("noseSneerLeft", "noseSneerRight"),
]


class VOCARKitDataset(ABC, Dataset):
    """Abstract class of VOCA-ARKit dataset"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        sampling_rate: int,
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
        """
        self.audio_dir = audio_dir
        self.blendshape_coeffs_dir = blendshape_coeffs_dir
        self.sampling_rate = sampling_rate

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

    @abstractmethod
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
                "blendshape_coeffs": (blendshape_seq_len, num_blendshapes),
            }
        """
        pass

    @staticmethod
    def collate_fn(examples: List[Dict[str, torch.FloatTensor]]) -> Dict[str, Any]:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[Dict[str, torch.FloatTensor]]
            List of the outputs of __getitem__

        Returns
        -------
        Dict[str, Any]
            {
                "waveform": List[np.ndarray], each: (audio_seq_len,)
                "blendshape_coeffs": torch.FloatTensor, (Batch, blendshape_seq_len, num_blendshapes)
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


class VOCARKitTrainDataset(VOCARKitDataset):
    """Train dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        sampling_rate: int,
        window_size: int = 120,
        fps: int = 60,
        uncond_prob: float = 0.1,
        hflip: bool = True,
        classes: List[str] = VOCARKIT_CLASSES,
        classes_mirror_pair: List[Tuple[str, str]] = VOCARKIT_CLASSES_MIRROR_PAIR,
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
        uncond_prob : float, optional
            Unconditional probability of waveform (for classifier-free guidance), by default 0.1
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        classes : List[str], optional
            List of blendshape names, by default VOCARKIT_CLASSES
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default VOCARKIT_CLASSES_MIRROR_PAIR
        """
        super().__init__(audio_dir, blendshape_coeffs_dir, sampling_rate)
        self.window_size = window_size
        self.fps = fps
        self.uncond_prob = uncond_prob

        self.hflip = hflip
        self.classes = classes
        self.classes_mirror_pair = classes_mirror_pair

        self.mirror_indices = []
        self.mirror_indices_flip = []
        for pair in self.classes_mirror_pair:
            index_l = self.classes.index(pair[0])
            index_r = self.classes.index(pair[1])
            self.mirror_indices.extend([index_l, index_r])
            self.mirror_indices_flip.extend([index_r, index_l])

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        waveform = load_audio(self.audio_paths[index], self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(self.blendshape_coeffs_paths[index])

        num_blendshape = blendshape_coeffs.shape[1]
        blendshape_len = blendshape_coeffs.shape[0]
        waveform_len = waveform.shape[0]
        waveform_patch_len = (self.sampling_rate * self.window_size) // self.fps

        waveform_patch = None
        blendshape_coeffs_patch = None
        if blendshape_len >= self.window_size:
            idx = random.randint(0, blendshape_len - self.window_size)
            blendshape_coeffs_patch = blendshape_coeffs[idx : idx + self.window_size, :]

            waveform_patch_idx = (self.sampling_rate * idx) // self.fps
            waveform_patch = waveform[
                waveform_patch_idx : waveform_patch_idx + waveform_patch_len
            ]
        else:
            blendshape_coeffs_patch = torch.zeros((self.window_size, num_blendshape))
            blendshape_coeffs_patch[:blendshape_len, :] = blendshape_coeffs[:, :]

            waveform_patch = torch.zeros(waveform_patch_len)
            waveform_patch[:waveform_len] = waveform[:]

        # Augmentation - hflip
        if self.hflip and random.uniform(0, 1) < 0.5:
            blendshape_coeffs_patch[:, self.mirror_indices] = blendshape_coeffs_patch[
                :, self.mirror_indices_flip
            ]

        # Random uncondition for classifier-free guidance
        if random.uniform(0, 1) < self.uncond_prob:
            waveform_patch = torch.zeros(waveform_patch_len)

        out = {
            "waveform": waveform_patch,
            "blendshape_coeffs": blendshape_coeffs_patch,
        }

        return out


class VOCARKitValDataset(VOCARKitDataset):
    """Validation dataset for VOCA-ARKit"""

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        waveform = load_audio(self.audio_paths[index], self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(self.blendshape_coeffs_paths[index])

        out = {
            "waveform": waveform,
            "blendshape_coeffs": blendshape_coeffs,
        }

        return out


class VOCARKitVAEDataset(Dataset):
    """Abstract class of VOCA-ARKit dataset for VAE"""

    def __init__(self, blendshape_coeffs_dir: str):
        """Constructor of the class

        Parameters
        ----------
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        """
        self.blendshape_coeffs_dir = blendshape_coeffs_dir

        self.blendshape_coeffs_paths = [
            os.path.join(self.blendshape_coeffs_dir, f)
            for f in sorted(os.listdir(self.blendshape_coeffs_dir))
        ]

        bl_seq_list = []
        for path in self.blendshape_coeffs_paths:
            bl_seq = load_blendshape_coeffs(path)
            bl_seq_list.append(bl_seq)

        self.blendshapes = torch.cat(bl_seq_list, dim=0)

        self.length = self.blendshapes.shape[0]

    def __len__(self) -> int:
        """Return the size of the dataset

        Returns
        -------
        int
            Size of the dataset
        """
        return self.length

    def __getitem__(self, index: int) -> torch.FloatTensor:
        """Return the item of the given index

        Parameters
        ----------
        index : int
            Index of the item

        Returns
        -------
        torch.FloatTensor
            "blendshape_coeffs": (1, num_blendshapes),
        """
        return self.blendshapes[index].unsqueeze(0)


class VOCARKitPseudoGTOptDataset:
    """Dataset for generating pseudo-GT blendshape coefficients"""

    def __init__(
        self,
        neutrals_dir: str,
        blendshapes_dir: str,
        mesh_seqs_dir: str,
        blendshapes_names: List[str],
        seq_id_range: Tuple[int, int] = (1, 40),
    ) -> None:
        self.neutrals_dir = neutrals_dir
        self.blendshapes_dir_dir = blendshapes_dir
        self.mesh_seqs_dir_dir_dir = mesh_seqs_dir
        self.blendshapes_names = blendshapes_names

        self.person_ids = sorted(os.listdir(self.mesh_seqs_dir_dir_dir))

        self.seq_ids = list(range(seq_id_range[0], seq_id_range[1] + 1))

    def get_person_id_list(self) -> List[str]:
        return self.person_ids

    def get_seq_id_list(self) -> List[int]:
        return self.seq_ids

    def get_blendshapes(
        self, person_id: str
    ) -> Dict[str, Union[trimesh.base.Trimesh, Dict[str, trimesh.base.Trimesh]]]:
        neutral_path = os.path.join(self.neutrals_dir, f"{person_id}.obj")
        blendshapes_dir = os.path.join(self.blendshapes_dir_dir, person_id)

        neutral_mesh = trimesh.load(neutral_path)

        blendshapes_dict = {}
        for bl_name in self.blendshapes_names:
            bl_path = os.path.join(blendshapes_dir, f"{bl_name}.obj")
            bl_mesh = trimesh.load(bl_path)
            blendshapes_dict[bl_name] = bl_mesh

        output = {}
        output["neutral"] = neutral_mesh
        output["blendshapes"] = blendshapes_dict

        return output

    def get_mesh_seq(self, person_id: str, seq_id: int) -> List[trimesh.base.Trimesh]:
        mesh_seq_dir = os.path.join(
            self.mesh_seqs_dir_dir_dir, person_id, f"sentence{seq_id:02}"
        )

        if not os.path.isdir(mesh_seq_dir):
            return []

        mesh_seq_paths = sorted(
            glob.glob(os.path.join(mesh_seq_dir, "**/*.obj"), recursive=True)
        )

        mesh_seq_list = []
        for seq_path in mesh_seq_paths:
            mesh = trimesh.load(seq_path)
            mesh_seq_list.append(mesh)

        return mesh_seq_list
