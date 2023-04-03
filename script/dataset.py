"""Define the dataset which are used in the scripts
"""
from abc import abstractmethod, ABC
import glob
import os
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import trimesh
from said.util.audio import load_audio
from said.util.blendshape import load_blendshape_coeffs, load_blendshape_deltas
from said.util.mesh import create_mesh, get_submesh, load_mesh
from said.util.parser import parse_list


class VOCARKitDataset(ABC, Dataset):
    """Abstract class of VOCA-ARKit dataset"""

    person_ids_train = [
        "FaceTalk_170725_00137_TA",
        "FaceTalk_170728_03272_TA",
        "FaceTalk_170811_03274_TA",
        "FaceTalk_170904_00128_TA",
        "FaceTalk_170904_03276_TA",
        "FaceTalk_170912_03278_TA",
        "FaceTalk_170913_03279_TA",
        "FaceTalk_170915_00223_TA",
    ]

    person_ids_val = [
        "FaceTalk_170811_03275_TA",
        "FaceTalk_170908_03277_TA",
    ]

    person_ids_test = [
        "FaceTalk_170731_00024_TA",
        "FaceTalk_170809_00138_TA",
    ]

    sentence_ids = list(range(1, 41))

    fps = 60

    default_blendshape_classes = [
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

    default_blendshape_classes_mirror_pair = [
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

    @abstractmethod
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
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset

        Returns
        -------
        int
            Size of the dataset
        """
        pass

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

    def get_data_paths(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        person_ids: List[str],
    ) -> List[Dict[str, str]]:
        """Return the list of the data paths

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        person_ids : List[str]
            List of the person ids

        Returns
        -------
        List[Dict[str, str]]
            [{
                "audio": audio_path,
                "coeffs": coeffs_path,
                "person_id": person id,
            }]
        """
        data_paths = []

        for pid in person_ids:
            audio_id_dir = os.path.join(audio_dir, pid)
            coeffs_id_dir = os.path.join(blendshape_coeffs_dir, pid)

            for sid in self.sentence_ids:
                audio_path = os.path.join(audio_id_dir, f"sentence{sid:02}.wav")
                coeffs_path = os.path.join(coeffs_id_dir, f"sentence{sid:02}.csv")

                if os.path.exists(audio_path) and os.path.exists(coeffs_path):
                    data = {
                        "audio": audio_path,  # str
                        "coeffs": coeffs_path,  # str
                        "person_id": pid,  # str
                    }
                    data_paths.append(data)

        return data_paths

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

    @staticmethod
    def preprocess_blendshapes(
        templates_dir: str,
        blendshape_deltas_path: str,
        blendshape_indices: Optional[List[int]] = None,
        person_ids: Optional[List[str]] = None,
        blendshape_classes: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Union[trimesh.Trimesh, Dict[str, trimesh.Trimesh]]]]:
        """Preprocess the blendshapes

        Parameters
        ----------
        templates_dir : str
            Directory path of the templates
        blendshape_deltas_path : str
            Path of the blendshape deltas file
        blendshape_indices : Optional[List[int]], optional
            List of the blendshape indices, by default None
        person_ids : Optional[List[str]], optional
            List of the person ids, by default None
        blendshape_classes : Optional[List[str]], optional
            List of the blendshape classes, by default None

        Returns
        -------
        Dict[str, Dict[str, Union[trimesh.Trimesh, Dict[str, trimesh.Trimesh]]]]
            {
                <Person id>: {
                    "neutral": trimesh.Trimesh, neutral mesh object,
                    "blendshapes": Dict[str, trimesh.Trimesh], {
                        <Blendshape name>: trimesh.Trimesh, blendshape mesh object
                    }
                }
            }
        """
        if blendshape_indices is None:
            blendshape_indices_path = (
                pathlib.Path(__file__).parent.parent / "data" / "FLAME_head_idx.txt"
            )
            blendshape_indices = parse_list(blendshape_indices_path, int)

        if person_ids is None:
            person_ids = (
                VOCARKitDataset.person_ids_train
                + VOCARKitDataset.person_ids_val
                + VOCARKitDataset.person_ids_test
            )

        if blendshape_classes is None:
            blendshape_classes = VOCARKitDataset.default_blendshape_classes

        blendshape_deltas = load_blendshape_deltas(blendshape_deltas_path)

        blendshapes = {}
        for pid in tqdm(person_ids):
            template_mesh_path = os.path.join(templates_dir, f"{pid}.ply")
            template_mesh_ori = load_mesh(template_mesh_path)
            submesh_out = get_submesh(
                template_mesh_ori.vertices, template_mesh_ori.faces, blendshape_indices
            )

            vertices = submesh_out["vertices"]
            faces = submesh_out["faces"]

            neutral_mesh = create_mesh(vertices, faces)

            bl_deltas = blendshape_deltas[pid]

            blendshapes_id = {}
            for bl_name in blendshape_classes:
                bl_vertices = vertices + bl_deltas[bl_name]
                blendshapes_id[bl_name] = create_mesh(bl_vertices, faces)

            blendshapes[pid] = {"neutral": neutral_mesh, "blendshapes": blendshapes_id}

        return blendshapes


class VOCARKitTrainDataset(VOCARKitDataset):
    """Train dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        sampling_rate: int,
        window_size: int = 120,
        uncond_prob: float = 0.1,
        hflip: bool = True,
        classes: List[str] = VOCARKitDataset.default_blendshape_classes,
        classes_mirror_pair: List[
            Tuple[str, str]
        ] = VOCARKitDataset.default_blendshape_classes_mirror_pair,
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
        uncond_prob : float, optional
            Unconditional probability of waveform (for classifier-free guidance), by default 0.1
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
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

        self.data_paths = self.get_data_paths(
            audio_dir, blendshape_coeffs_dir, self.person_ids_train
        )

        self.waveform_window_len = (self.sampling_rate * self.window_size) // self.fps

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        data = self.data_paths[index]
        waveform = load_audio(data["audio"], self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(data["coeffs"])

        num_blendshape = blendshape_coeffs.shape[1]
        blendshape_len = blendshape_coeffs.shape[0]

        # Random-select the window
        bdx = random.randint(0, max(0, blendshape_len - self.window_size))
        wdx = (self.sampling_rate * bdx) // self.fps

        waveform_tmp = waveform[wdx : wdx + self.waveform_window_len]
        coeffs_tmp = blendshape_coeffs[bdx : bdx + self.window_size, :]

        waveform_window = torch.zeros(self.waveform_window_len)
        coeffs_window = torch.zeros((self.window_size, num_blendshape))

        waveform_window[: waveform_tmp.shape[0]] = waveform_tmp[:]
        coeffs_window[: coeffs_tmp.shape[0], :] = coeffs_tmp[:]

        # Augmentation - hflip
        if self.hflip and random.uniform(0, 1) < 0.5:
            coeffs_window[:, self.mirror_indices] = coeffs_window[
                :, self.mirror_indices_flip
            ]

        # Random uncondition for classifier-free guidance
        if random.uniform(0, 1) < self.uncond_prob:
            waveform_window = torch.zeros(self.waveform_window_len)

        out = {
            "waveform": waveform_window,
            "blendshape_coeffs": coeffs_window,
        }

        return out


class VOCARKitValDataset(VOCARKitDataset):
    """Validation dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        sampling_rate: int,
        uncond_prob: float = 0.1,
        hflip: bool = True,
        classes: List[str] = VOCARKitDataset.default_blendshape_classes,
        classes_mirror_pair: List[
            Tuple[str, str]
        ] = VOCARKitDataset.default_blendshape_classes_mirror_pair,
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
        uncond_prob : float, optional
            Unconditional probability of waveform (for classifier-free guidance), by default 0.1
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        """
        self.sampling_rate = sampling_rate
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

        self.data_paths = self.get_data_paths(
            audio_dir, blendshape_coeffs_dir, self.person_ids_val
        )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.FloatTensor]:
        data = self.data_paths[index]
        waveform = load_audio(data["audio"], self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(data["coeffs"])

        waveform_len = waveform.shape[0]
        blendshape_len = blendshape_coeffs.shape[0]
        waveform_window_len = (self.sampling_rate * blendshape_len) // self.fps

        # Adjust the waveform window
        waveform_tmp = waveform[:waveform_window_len]

        waveform_window = torch.zeros(waveform_window_len)
        waveform_window[: waveform_tmp.shape[0]] = waveform_tmp[:]

        out = {
            "waveform": waveform_window,
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
    ) -> None:
        """Constructor of the VOCARKitPseudoGTOptDataset

        Parameters
        ----------
        neutrals_dir : str
            Directory which contains the neutral meshes
        blendshapes_dir : str
            Directory which contains the blendshape meshes
        mesh_seqs_dir : str
            Directory which contains the mesh sequences
        blendshapes_names : List[str]
            List of the blendshape names
        """
        self.neutrals_dir = neutrals_dir
        self.blendshapes_dir_dir = blendshapes_dir
        self.mesh_seqs_dir_dir_dir = mesh_seqs_dir
        self.blendshapes_names = blendshapes_names

    def get_blendshapes(
        self, person_id: str
    ) -> Dict[str, Union[trimesh.base.Trimesh, Dict[str, trimesh.base.Trimesh]]]:
        """Return the dictionary of the blendshape meshes

        Parameters
        ----------
        person_id : str
            Person id that wants to get the blendshapes

        Returns
        -------
        Dict[str, Union[trimesh.base.Trimesh, Dict[str, trimesh.base.Trimesh]]]
            {
                "neutral": trimesh.base.Trimesh, neutral mesh
                "blendshape": Dict[str, trimesh.base.Trimesh], dictionary of the blendshape meshes
                    {
                        "{blendshape name}": trimesh.base.Trimesh, each blendshape meshes
                    }
            }
        """
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
        """Return the mesh sequence

        Parameters
        ----------
        person_id : str
            Person id
        seq_id : int
            Sequence id

        Returns
        -------
        List[trimesh.base.Trimesh]
            List of the meshes
        """
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
