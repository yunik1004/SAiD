"""Define the dataset which are used in the scripts
"""
from abc import abstractmethod, ABC
import glob
from dataclasses import dataclass
import os
import pathlib
import random
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import trimesh
from said.util.audio import load_audio
from said.util.blendshape import load_blendshape_coeffs, load_blendshape_deltas
from said.util.mesh import create_mesh, get_submesh, load_mesh
from said.util.parser import parse_list


@dataclass
class DataItem:
    """Dataclass for the data item"""

    waveform: Optional[torch.FloatTensor]  # (audio_seq_len,)
    blendshape_coeffs: Optional[
        torch.FloatTensor
    ]  # (blendshape_seq_len, num_blendshapes)
    cond: bool = True  # If false, uncondition for classifier-free guidance
    blendshape_delta: Optional[torch.FloatTensor] = None  # (num_blendshapes, |V|, 3)
    person_id: Optional[str] = None
    sentence_id: Optional[int] = None


@dataclass
class DataBatch:
    """Dataclass for the data batch"""

    waveform: List[np.ndarray]  # List[(audio_seq_len,)] with length batch_size
    blendshape_coeffs: Optional[
        torch.FloatTensor
    ]  # (batch_size, blendshape_seq_len, num_blendshapes)
    cond: torch.BoolTensor  # (batch_size,)
    blendshape_delta: Optional[
        torch.FloatTensor
    ] = None  # (batch_size, num_blendshapes, |V|, 3)
    person_ids: Optional[List[str]] = None
    sentence_ids: Optional[List[int]] = None


@dataclass
class ExpressionBases:
    """Dataclass for the expression bases (including neutral)"""

    neutral: trimesh.Trimesh  # Neutral mesh object
    blendshapes: Dict[str, trimesh.Trimesh]  # <blendshape name>: blendshape mesh object


@dataclass
class VOCARKitDataPath:
    """Dataclass for the VOCARKit data path"""

    person_id: str
    sentence_id: int
    audio: Optional[str]
    blendshape_coeffs: Optional[str]


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
    def __len__(self) -> int:
        """Return the size of the dataset

        Returns
        -------
        int
            Size of the dataset
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> DataItem:
        """Return the item of the given index

        Parameters
        ----------
        index : int
            Index of the item

        Returns
        -------
        DataItem
            DataItem object
        """
        pass

    def get_data_paths(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: Optional[str],
        person_ids: List[str],
    ) -> List[VOCARKitDataPath]:
        """Return the list of the data paths

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : Optional[str]
            Directory of the blendshape coefficients
        person_ids : List[str]
            List of the person ids

        Returns
        -------
        List[VOCARKitDataPath]
            List of the VOCARKitDataPath objects
        """
        data_paths = []

        for pid in person_ids:
            audio_id_dir = os.path.join(audio_dir, pid)
            coeffs_id_dir = (
                os.path.join(blendshape_coeffs_dir, pid)
                if blendshape_coeffs_dir
                else None
            )

            for sid in self.sentence_ids:
                filename_base = f"sentence{sid:02}"
                audio_path = os.path.join(audio_id_dir, f"{filename_base}.wav")

                if not os.path.exists(audio_path):
                    continue

                if coeffs_id_dir and os.path.exists(coeffs_id_dir):
                    coeffs_pattern = re.compile(f"^{filename_base}(-.+)?\.csv$")
                    filename_list = [
                        s for s in os.listdir(coeffs_id_dir) if coeffs_pattern.match(s)
                    ]
                    for filename in filename_list:
                        coeffs_path = os.path.join(coeffs_id_dir, filename)
                        if os.path.exists(coeffs_path):
                            data = VOCARKitDataPath(
                                person_id=pid,
                                sentence_id=sid,
                                audio=audio_path,
                                blendshape_coeffs=coeffs_path,
                            )
                            data_paths.append(data)
                else:
                    data = VOCARKitDataPath(
                        person_id=pid,
                        sentence_id=sid,
                        audio=audio_path,
                        blendshape_coeffs=None,
                    )
                    data_paths.append(data)

        return data_paths

    @staticmethod
    def collate_fn(examples: List[DataItem]) -> DataBatch:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[DataItem]
            List of the outputs of __getitem__

        Returns
        -------
        DataBatch
            DataBatch object
        """
        waveforms = [np.array(item.waveform) for item in examples]
        blendshape_coeffss = None
        if len(examples) > 0 and examples[0].blendshape_coeffs is not None:
            blendshape_coeffss = torch.stack(
                [item.blendshape_coeffs for item in examples]
            )
        conds = torch.BoolTensor([item.cond for item in examples])
        blendshape_deltas = None
        if len(examples) > 0 and examples[0].blendshape_delta is not None:
            blendshape_deltas = torch.stack(
                [item.blendshape_delta for item in examples]
            )

        person_ids = None
        if len(examples) > 0 and examples[0].person_id is not None:
            person_ids = [item.person_id for item in examples]

        sentence_ids = None
        if len(examples) > 0 and examples[0].sentence_id is not None:
            sentence_ids = [item.sentence_id for item in examples]

        return DataBatch(
            waveform=waveforms,
            blendshape_coeffs=blendshape_coeffss,
            cond=conds,
            blendshape_delta=blendshape_deltas,
            person_ids=person_ids,
            sentence_ids=sentence_ids,
        )

    @staticmethod
    def preprocess_blendshapes(
        templates_dir: str,
        blendshape_deltas_path: str,
        blendshape_indices: Optional[List[int]] = None,
        person_ids: Optional[List[str]] = None,
        blendshape_classes: Optional[List[str]] = None,
    ) -> Dict[str, ExpressionBases]:
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
        Dict[str, ExpressionBases]
            {
                <Person id>: expression bases
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

        expressions = {}
        for pid in tqdm(person_ids):
            template_mesh_path = os.path.join(templates_dir, f"{pid}.ply")
            template_mesh_ori = load_mesh(template_mesh_path)
            submesh_out = get_submesh(
                template_mesh_ori.vertices, template_mesh_ori.faces, blendshape_indices
            )

            vertices = submesh_out.vertices
            faces = submesh_out.faces

            neutral_mesh = create_mesh(vertices, faces)

            bl_deltas = blendshape_deltas[pid]

            blendshapes_dict = {}
            for bl_name in blendshape_classes:
                bl_vertices = vertices + bl_deltas[bl_name]
                blendshapes_dict[bl_name] = create_mesh(bl_vertices, faces)

            expressions[pid] = ExpressionBases(
                neutral=neutral_mesh, blendshapes=blendshapes_dict
            )

        return expressions


class VOCARKitTrainDataset(VOCARKitDataset):
    """Train dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        blendshape_deltas_path: Optional[str],
        sampling_rate: int,
        window_size: int = 120,
        uncond_prob: float = 0.1,
        zero_prob: float = 0.003125,
        hflip: bool = True,
        delay: bool = True,
        delay_thres: int = 1,
        classes: List[str] = VOCARKitDataset.default_blendshape_classes,
        classes_mirror_pair: List[
            Tuple[str, str]
        ] = VOCARKitDataset.default_blendshape_classes_mirror_pair,
    ) -> None:
        """Constructor of the class

        Parameters
        ----------
        audio_dir : str
            Directory of the audio data
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        blendshape_deltas_path : Optional[str]
            Path of the blendshape deltas
        sampling_rate : int
            Sampling rate of the audio
        window_size : int, optional
            Window size of the blendshape coefficients, by default 120
        uncond_prob : float, optional
            Unconditional probability of waveform (for classifier-free guidance), by default 0.1
        zero_prob : float, optional
            Zero-out probability of waveform and blendshape coefficients, by default 0.003125
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        delay : bool, optional
            Whether do the delaying, by default True
        delay_thres: int, optional
            Maximum amount of delaying, by default 1
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.uncond_prob = uncond_prob
        self.zero_prob = zero_prob

        self.hflip = hflip
        self.delay = delay
        self.delay_thres = delay_thres
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

        self.blendshape_deltas = (
            load_blendshape_deltas(blendshape_deltas_path)
            if blendshape_deltas_path
            else None
        )

        self.waveform_window_len = (self.sampling_rate * self.window_size) // self.fps

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]
        waveform = load_audio(data.audio, self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(data.blendshape_coeffs)
        blendshape_delta = (
            torch.FloatTensor(
                np.stack(list(self.blendshape_deltas[data.person_id].values()), axis=0)
            )
            if self.blendshape_deltas
            else None
        )

        num_blendshape = blendshape_coeffs.shape[1]
        blendshape_len = blendshape_coeffs.shape[0]

        # Random-select the window
        bdx = random.randint(0, max(0, blendshape_len - self.window_size))
        wdx = (self.sampling_rate * bdx) // self.fps
        if self.delay and random.uniform(0, 1) < 0.5:
            wdelays = list(range(max(wdx - self.delay_thres, 0), wdx)) + list(
                range(wdx + 1, wdx + self.delay_thres + 1)
            )
            wdx = random.choice(wdelays)

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
        cond = random.uniform(0, 1) > self.uncond_prob

        # Random zero-out
        if random.uniform(0, 1) < self.zero_prob:
            waveform_window = torch.zeros_like(waveform_window)
            coeffs_window = torch.zeros_like(coeffs_window)

        return DataItem(
            waveform=waveform_window,
            blendshape_coeffs=coeffs_window,
            cond=cond,
            blendshape_delta=blendshape_delta,
        )


class VOCARKitValDataset(VOCARKitDataset):
    """Validation dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        blendshape_deltas_path: Optional[str],
        sampling_rate: int,
        uncond_prob: float = 0.1,
        zero_prob: float = 0.003125,
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
        blendshape_deltas_path : Optional[str]
            Path of the blendshape deltas
        sampling_rate : int
            Sampling rate of the audio
        uncond_prob : float, optional
            Unconditional probability of waveform (for classifier-free guidance), by default 0.1
        zero_prob : float, optional
            Zero-out probability of waveform and blendshape coefficients, by default 0.003125
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        """
        self.sampling_rate = sampling_rate
        self.uncond_prob = uncond_prob
        self.zero_prob = zero_prob

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

        self.blendshape_deltas = (
            load_blendshape_deltas(blendshape_deltas_path)
            if blendshape_deltas_path
            else None
        )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]
        waveform = load_audio(data.audio, self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(data.blendshape_coeffs)
        blendshape_delta = (
            torch.FloatTensor(
                np.stack(list(self.blendshape_deltas[data.person_id].values()), axis=0)
            )
            if self.blendshape_deltas
            else None
        )

        blendshape_len = blendshape_coeffs.shape[0]
        waveform_window_len = (self.sampling_rate * blendshape_len) // self.fps

        # Adjust the waveform window
        waveform_tmp = waveform[:waveform_window_len]

        waveform_window = torch.zeros(waveform_window_len)
        waveform_window[: waveform_tmp.shape[0]] = waveform_tmp[:]

        # Random uncondition for classifier-free guidance
        cond = random.uniform(0, 1) > self.uncond_prob

        # Random zero-out
        if random.uniform(0, 1) < self.zero_prob:
            waveform_window = torch.zeros_like(waveform_window)
            blendshape_coeffs = torch.zeros_like(blendshape_coeffs)

        return DataItem(
            waveform=waveform_window,
            blendshape_coeffs=blendshape_coeffs,
            cond=cond,
            blendshape_delta=blendshape_delta,
        )


class VOCARKitTestDataset(VOCARKitDataset):
    """Test dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: Optional[str],
        blendshape_deltas_path: Optional[str],
        sampling_rate: int,
    ):
        self.sampling_rate = sampling_rate

        self.data_paths = self.get_data_paths(
            audio_dir, blendshape_coeffs_dir, self.person_ids_test
        )

        self.blendshape_deltas = (
            load_blendshape_deltas(blendshape_deltas_path)
            if blendshape_deltas_path
            else None
        )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]
        waveform = load_audio(data.audio, self.sampling_rate)
        blendshape_coeffs = (
            load_blendshape_coeffs(data.blendshape_coeffs)
            if data.blendshape_coeffs
            else None
        )
        blendshape_delta = (
            torch.FloatTensor(
                np.stack(list(self.blendshape_deltas[data.person_id].values()), axis=0)
            )
            if self.blendshape_deltas
            else None
        )

        waveform_window = waveform
        if blendshape_coeffs is not None:
            blendshape_len = blendshape_coeffs.shape[0]
            waveform_window_len = (self.sampling_rate * blendshape_len) // self.fps

            # Adjust the waveform window
            waveform_tmp = waveform[:waveform_window_len]

            waveform_window = torch.zeros(waveform_window_len)
            waveform_window[: waveform_tmp.shape[0]] = waveform_tmp[:]

        return DataItem(
            waveform=waveform_window,
            blendshape_coeffs=blendshape_coeffs,
            blendshape_delta=blendshape_delta,
        )


class VOCARKitEvalDataset(VOCARKitDataset):
    """Evaluation dataset for VOCA-ARKit"""

    def __init__(
        self,
        audio_dir: str,
        blendshape_coeffs_dir: str,
        blendshape_deltas_path: Optional[str],
        sampling_rate: int,
        classes: List[str] = VOCARKitDataset.default_blendshape_classes,
    ):
        self.sampling_rate = sampling_rate
        self.classes = classes

        self.data_paths = self.get_data_paths(
            audio_dir, blendshape_coeffs_dir, self.person_ids_test
        )

        self.blendshape_deltas = (
            load_blendshape_deltas(blendshape_deltas_path)
            if blendshape_deltas_path
            else None
        )

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]
        waveform = load_audio(data.audio, self.sampling_rate)
        blendshape_coeffs = load_blendshape_coeffs(data.blendshape_coeffs)
        blendshape_delta = (
            torch.FloatTensor(
                np.stack(list(self.blendshape_deltas[data.person_id].values()), axis=0)
            )
            if self.blendshape_deltas
            else None
        )

        blendshape_len = blendshape_coeffs.shape[0]
        waveform_window_len = (self.sampling_rate * blendshape_len) // self.fps

        # Adjust the waveform window
        waveform_tmp = waveform[:waveform_window_len]

        waveform_window = torch.zeros(waveform_window_len)
        waveform_window[: waveform_tmp.shape[0]] = waveform_tmp[:]

        return DataItem(
            waveform=waveform_window,
            blendshape_coeffs=blendshape_coeffs,
            blendshape_delta=blendshape_delta,
            person_id=data.person_id,
            sentence_id=data.sentence_id,
        )


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

    def get_blendshapes(self, person_id: str) -> ExpressionBases:
        """Return the dictionary of the blendshape meshes

        Parameters
        ----------
        person_id : str
            Person id that wants to get the blendshapes

        Returns
        -------
        ExpressionBases
            Expression bases object
        """
        neutral_path = os.path.join(self.neutrals_dir, f"{person_id}.obj")
        blendshapes_dir = os.path.join(self.blendshapes_dir_dir, person_id)

        neutral_mesh = load_mesh(neutral_path)

        blendshapes_dict = {}
        for bl_name in self.blendshapes_names:
            bl_path = os.path.join(blendshapes_dir, f"{bl_name}.obj")
            bl_mesh = load_mesh(bl_path)
            blendshapes_dict[bl_name] = bl_mesh

        return ExpressionBases(neutral=neutral_mesh, blendshapes=blendshapes_dict)

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
            mesh = load_mesh(seq_path)
            mesh_seq_list.append(mesh)

        return mesh_seq_list


class VOCARKitVAEDataset(VOCARKitDataset):
    """Abstract class of VOCA-ARKit dataset for VAE"""

    def __init__(
        self,
        blendshape_coeffs_dir: str,
        window_size: int = 120,
        zero_prob: float = 0.003125,
        hflip: bool = True,
        dataset_type: str = "train",
        classes: List[str] = VOCARKitDataset.default_blendshape_classes,
        classes_mirror_pair: List[
            Tuple[str, str]
        ] = VOCARKitDataset.default_blendshape_classes_mirror_pair,
    ) -> None:
        """Constructor of the class

        Parameters
        ----------
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        window_size : int, optional
            Window size of the blendshape coefficients, by default 120
        zero_prob : float, optional
            Zero-out probability of waveform and blendshape coefficients, by default 0.003125
        hflip : bool, optional
            Whether do the horizontal flip, by default True
        dataset_type: str, optional
            Type of the dataset, whether "train", "val", and "test", by default "train"
        classes : List[str], optional
            List of blendshape names, by default default_blendshape_classes
        classes_mirror_pair : List[Tuple[str, str]], optional
            List of blendshape pairs which are mirror to each other, by default default_blendshape_classes_mirror_pair
        """
        self.window_size = window_size
        self.zero_prob = zero_prob

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

        person_ids = None
        if dataset_type == "train":
            person_ids = self.person_ids_train
        elif dataset_type == "val":
            person_ids = self.person_ids_val
        else:
            person_ids = self.person_ids_test

        self.data_paths = self.get_data_paths(blendshape_coeffs_dir, person_ids)

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, index: int) -> DataItem:
        data = self.data_paths[index]
        blendshape_coeffs = load_blendshape_coeffs(data.blendshape_coeffs)

        num_blendshape = blendshape_coeffs.shape[1]
        blendshape_len = blendshape_coeffs.shape[0]

        # Random-select the window
        bdx = random.randint(0, max(0, blendshape_len - self.window_size))
        coeffs_tmp = blendshape_coeffs[bdx : bdx + self.window_size, :]
        coeffs_window = torch.zeros((self.window_size, num_blendshape))
        coeffs_window[: coeffs_tmp.shape[0], :] = coeffs_tmp[:]

        # Augmentation - hflip
        if self.hflip and random.uniform(0, 1) < 0.5:
            coeffs_window[:, self.mirror_indices] = coeffs_window[
                :, self.mirror_indices_flip
            ]

        # Random zero-out
        if random.uniform(0, 1) < self.zero_prob:
            coeffs_window = torch.zeros_like(coeffs_window)

        return DataItem(
            waveform=None,
            blendshape_coeffs=coeffs_window,
        )

    def get_data_paths(
        self,
        blendshape_coeffs_dir: str,
        person_ids: List[str],
    ) -> List[VOCARKitDataPath]:
        """Return the list of the data paths

        Parameters
        ----------
        blendshape_coeffs_dir : str
            Directory of the blendshape coefficients
        person_ids : List[str]
            List of the person ids

        Returns
        -------
        List[VOCARKitDataPath]
            List of the VOCARKitDataPath objects
        """
        data_paths = []

        for pid in person_ids:
            coeffs_id_dir = os.path.join(blendshape_coeffs_dir, pid)
            if coeffs_id_dir is None or not os.path.exists(coeffs_id_dir):
                continue

            for sid in self.sentence_ids:
                filename_base = f"sentence{sid:02}"
                coeffs_pattern = re.compile(f"^{filename_base}(-.+)?\.csv$")
                filename_list = [
                    s for s in os.listdir(coeffs_id_dir) if coeffs_pattern.match(s)
                ]
                for filename in filename_list:
                    coeffs_path = os.path.join(coeffs_id_dir, filename)
                    if os.path.exists(coeffs_path):
                        data = VOCARKitDataPath(
                            person_id=pid,
                            sentence_id=sid,
                            audio=None,
                            blendshape_coeffs=coeffs_path,
                        )
                        data_paths.append(data)

        return data_paths

    @staticmethod
    def collate_fn(examples: List[DataItem]) -> DataBatch:
        """Collate function which is used for dataloader

        Parameters
        ----------
        examples : List[DataItem]
            List of the outputs of __getitem__

        Returns
        -------
        DataBatch
            DataBatch object
        """
        blendshape_coeffss = None
        if len(examples) > 0 and examples[0].blendshape_coeffs is not None:
            blendshape_coeffss = torch.stack(
                [item.blendshape_coeffs for item in examples]
            )
        conds = torch.BoolTensor([item.cond for item in examples])

        return DataBatch(
            waveform=[],
            blendshape_coeffs=blendshape_coeffss,
            cond=conds,
        )
