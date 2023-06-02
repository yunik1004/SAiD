"""Evaluate the data
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from said.metric.beat_consistency import beat_consistency_score
from said.metric.frechet_distance import frechet_distance, get_statistic
from said.metric.multimodality import multimodality
from said.model.vae import BCVAE
from dataset.dataset_voca import VOCARKitEvalDataset


@dataclass
class EvalMetrics:
    """
    Dataclass for the evaluation metrics
    """

    beat_consistency_score: float
    frechet_distance: float
    multimodality: float


@dataclass
class LatentInfo:
    """
    Dataclass for the latent
    """

    person_id: str
    sentence_id: int
    frame_start: int
    latent: np.ndarray


def generate_latents_info(
    vae: BCVAE,
    dataloader: DataLoader,
    device: Union[str, torch.device],
    window_step_size: int,
    padding: int = 0,
) -> List[LatentInfo]:
    """
    Generate the latent infos

    Parameters
    ----------
    vae: BCVAE
        VAE object
    dataloader: DataLoader
        Data loader
    device: Union[str, torch.device],
        GPU/CPU device
    window_step_size: int
        Step of the window movements
    padding: int, optional
        Amount of right padding applied to the input, by default 0

    Returns
    -------
    List[LatentInfo]
        Latent infos of dataset
    """
    latents_info = []

    with torch.no_grad():
        for data in dataloader:
            person_id = data.person_ids[0]
            sentence_id = data.sentence_ids[0]
            coeffs = data.blendshape_coeffs.to(device)

            coeffs_len = coeffs.shape[1]
            num_windows = (coeffs_len - vae.seq_len) // window_step_size + 1 - padding

            for wdx in range(num_windows):
                frame_start = window_step_size * wdx
                coeffs_window = coeffs[:, frame_start : frame_start + vae.seq_len]
                latent = vae.encode(coeffs_window).mean[0].cpu().numpy()

                latent_info = LatentInfo(
                    person_id=person_id,
                    sentence_id=sentence_id,
                    frame_start=frame_start,
                    latent=latent,
                )

                latents_info.append(latent_info)

    return latents_info


def filter_latent_infos(
    eval_latents_info: List[LatentInfo],
    real_latents_info: List[LatentInfo],
) -> List[LatentInfo]:
    """
    Filter the eval latents infos by remaining the infos with person id, sentence id, frame start overlapped

    Parameters
    ----------
    eval_latents_info: List[LatentInfo]
        Latent infos in evaluation dataset
    real_latents_info: List[LatentInfo]
        Latent infos in real dataset

    Returns
    -------
    List[LatentInfo]:
        Filtered latent infos of evaluation dataset
    """
    real_latents_groups = {
        (info.person_id, info.sentence_id, info.frame_start)
        for info in real_latents_info
    }

    eval_latents_info_filtered = [
        info
        for info in eval_latents_info
        if (info.person_id, info.sentence_id, info.frame_start) in real_latents_groups
    ]

    return eval_latents_info_filtered


def evaluate_beat_consistency_score(
    eval_dataloader: DataLoader,
    real_dataloader: DataLoader,
    sampling_rate: int,
    fps: int,
    bc_threshold: float,
) -> float:
    """Evaluate the beat consistency score

    Parameters
    ----------
    eval_dataloader: DataLoader
        Dataloader of the eval dataset
    real_dataloader: DataLoader
        Dataloader of the real dataset. It is required to remove eval data which is not in real dataset.
    sampling_rate: int
        Sampling rate of the waveform
    fps: int
        FPS of the blendshape coefficients sequence
    bc_threshold: float
        Threshold for computing beat consistency score

    Returns
    -------
    float
        Beat consistency score
    """
    real_data_groups = {
        (data.person_ids[0], data.sentence_ids[0]) for data in real_dataloader
    }

    list_waveform = []
    list_blendshape_coeffs = []
    for data in eval_dataloader:
        if (data.person_ids[0], data.sentence_ids[0]) in real_data_groups:
            waveform = data.waveform[0]
            coeffs = data.blendshape_coeffs[0].numpy()

            list_waveform.append(waveform)
            list_blendshape_coeffs.append(coeffs)

    return beat_consistency_score(
        list_waveform=list_waveform,
        list_blendshape_coeffs=list_blendshape_coeffs,
        sampling_rate=sampling_rate,
        fps=fps,
        threshold=bc_threshold,
    )


def evalute_frechet_distance(
    eval_latents_info: List[LatentInfo],
    real_latents_info: List[LatentInfo],
) -> float:
    """Evaluate the Frechet distance

    Parameters
    ----------
    eval_latents_info: List[LatentInfo]
        Latent infos in evaluation dataset
    real_latents_info: List[LatentInfo]
        Latent infos in real dataset

    Returns
    -------
    float
        Frechet distance
    """
    eval_latents = [info.latent for info in eval_latents_info]
    real_latents = [info.latent for info in real_latents_info]

    eval_statistic = get_statistic(eval_latents)
    real_statistic = get_statistic(real_latents)

    return frechet_distance(
        mu1=eval_statistic.mean,
        sigma1=eval_statistic.cov,
        mu2=real_statistic.mean,
        sigma2=real_statistic.cov,
    )


def evalute_multimodality(
    latents_info: List[LatentInfo],
) -> float:
    """Evaluate the multimodality

    Parameters
    ----------
    latents_info: List[LatentInfo]
        Latent infos

    Returns
    -------
    float
        Multimodality
    """
    latents_dict = defaultdict(list)
    for info in latents_info:
        latents_dict[(info.person_id, info.sentence_id, info.frame_start)].append(
            info.latent
        )

    latents_subset1 = []
    latents_subset2 = []

    for latents in latents_dict.values():
        half_len = len(latents) // 2

        latents_subset1.extend(latents[:half_len])
        latents_subset2.extend(latents[half_len : 2 * half_len])

    return multimodality(latents_subset1, latents_subset2)


def evaluate(
    eval_dataloader: DataLoader,
    real_dataloader: DataLoader,
    sampling_rate: int,
    fps: int,
    bc_threshold: float,
    vae: BCVAE,
    device: Union[str, torch.device],
    window_step_size: int,
) -> EvalMetrics:
    """Evaluate the data

    Parameters
    ----------
    eval_dataloader: DataLoader
        Dataloader of the VOCARKitEvalDataset
    real_dataloader: DataLoader
        Dataloader of the real dataset
    sampling_rate: int
        Sampling rate of the waveform
    fps: int
        FPS of the blendshape coefficients sequence
    bc_threshold: float
        Threshold for computing beat consistency score
    vae: BCVAE
        VAE object for generating latents
    device: Union[str, torch.device]
        GPU/CPU device
    window_step_size: int
        Step of the window movements for the latent generation

    Returns
    -------
    EvalMetrics
        Evaluation metrics
    """
    # Compute beat consistency score
    bc = evaluate_beat_consistency_score(
        eval_dataloader=eval_dataloader,
        real_dataloader=real_dataloader,
        sampling_rate=sampling_rate,
        fps=fps,
        bc_threshold=bc_threshold,
    )

    # Generate latents
    eval_latents_info = generate_latents_info(
        vae,
        eval_dataloader,
        device,
        window_step_size,
    )
    real_latents_info = generate_latents_info(
        vae,
        real_dataloader,
        device,
        window_step_size,
        padding=2,  # Padding is required to use same real dataset
    )

    # Filter the eval latents
    eval_latents_info_filtered = filter_latent_infos(
        eval_latents_info=eval_latents_info,
        real_latents_info=real_latents_info,
    )

    # Compute frechet distance
    fd = evalute_frechet_distance(
        eval_latents_info=eval_latents_info_filtered,
        real_latents_info=real_latents_info,
    )

    # Compute multimodality
    multimodality = evalute_multimodality(
        latents_info=eval_latents_info_filtered,
    )

    return EvalMetrics(
        beat_consistency_score=bc,
        frechet_distance=fd,
        multimodality=multimodality,
    )


def main() -> None:
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the output based on the VOCA-ARKit test dataset"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="../VOCA_ARKit/audio",
        help="Directory of the audio data",
    )
    parser.add_argument(
        "--coeffs_dir",
        type=str,
        default="../VOCA_ARKit/blendshape_coeffs",
        help="Directory of the blendshape coefficients data",
    )
    parser.add_argument(
        "--coeffs_real_dir",
        type=str,
        default="../VOCA_ARKit/blendshape_coeffs",
        help="Directory of the blendshape coefficients data",
    )
    parser.add_argument(
        "--vae_weights_path",
        type=str,
        default="../output/20000.pth",
        help="Path of the weights of VAE",
    )
    parser.add_argument(
        "--blendshape_deltas_path",
        type=str,
        default="../VOCA_ARKit/blendshape_deltas.pickle",
        help="Path of the blendshape deltas",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=16000,
        help="Sampling rate of the audio",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS of the blendshape coefficients sequence",
    )
    parser.add_argument(
        "--bc_threshold",
        type=float,
        default=0.1,
        help="Threshold for computing beat consistency score",
    )
    parser.add_argument(
        "--window_step_size",
        type=int,
        default=1,
        help="Step of the window movements for the latent generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU/CPU device",
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir
    coeffs_dir = args.coeffs_dir
    coeffs_real_dir = args.coeffs_real_dir
    vae_weights_path = args.vae_weights_path
    blendshape_deltas_path = None  # args.blendshape_deltas_path
    sampling_rate = args.sampling_rate
    fps = args.fps
    bc_threshold = args.bc_threshold
    window_step_size = args.window_step_size
    device = args.device

    # Load VAE
    said_vae = BCVAE()
    said_vae.load_state_dict(torch.load(vae_weights_path, map_location=device))
    said_vae.to(device)
    said_vae.eval()

    # Load data
    eval_dataset = VOCARKitEvalDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_dir,
        blendshape_deltas_path=blendshape_deltas_path,
        sampling_rate=sampling_rate,
    )

    real_dataset = VOCARKitEvalDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_real_dir,
        blendshape_deltas_path=blendshape_deltas_path,
        sampling_rate=sampling_rate,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=VOCARKitEvalDataset.collate_fn,
    )

    real_dataloader = DataLoader(
        real_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=VOCARKitEvalDataset.collate_fn,
    )

    # Evaluate the data
    eval_metrics = evaluate(
        eval_dataloader=eval_dataloader,
        real_dataloader=real_dataloader,
        sampling_rate=sampling_rate,
        fps=fps,
        bc_threshold=bc_threshold,
        vae=said_vae,
        device=device,
        window_step_size=window_step_size,
    )

    # Print the output
    print(eval_metrics)


if __name__ == "__main__":
    main()
