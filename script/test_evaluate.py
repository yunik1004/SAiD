"""Evaluate the data
"""
import argparse
from dataclasses import dataclass
from torch.utils.data import DataLoader
from said.util.audio import load_audio
from said.util.blendshape import load_blendshape_coeffs
from said.metric.beat_consistency import beat_consistency_score
from dataset.dataset_voca import VOCARKitEvalDataset


@dataclass
class EvalMetrics:
    """
    Dataclass for the evaluation metrics
    """

    beat_consistency_score: float


def evaluate(
    eval_dataloader: DataLoader, sampling_rate: int, fps: int, bc_threshold: float
) -> EvalMetrics:
    """Evaluate the data

    Parameters
    ----------
    eval_dataloader: DataLoader
        Dataloader of the VOCARKitEvalDataset
    sampling_rate: int
        Sampling rate of the waveform
    fps: int
        FPS of the blendshape coefficients sequence
    bc_threshold: float
        Threshold for computing beat consistency score

    Returns
    -------
    EvalMetrics
        Evaluation metrics
    """
    list_waveform = []
    list_blendshape_coeffs = []
    for data in eval_dataloader:
        waveform = data.waveform[0]
        coeffs = data.blendshape_coeffs[0].numpy()

        list_waveform.append(waveform)
        list_blendshape_coeffs.append(coeffs)

    bc = beat_consistency_score(
        list_waveform=list_waveform,
        list_blendshape_coeffs=list_blendshape_coeffs,
        sampling_rate=sampling_rate,
        fps=fps,
        threshold=bc_threshold,
    )

    return EvalMetrics(beat_consistency_score=bc)


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
        "--output_dir",
        type=str,
        default="../output",
        help="Directory of the outputs",
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir
    coeffs_dir = args.coeffs_dir
    blendshape_deltas_path = args.blendshape_deltas_path
    sampling_rate = args.sampling_rate
    fps = args.fps
    bc_threshold = args.bc_threshold

    output_dir = args.output_dir

    # Load data
    eval_dataset = VOCARKitEvalDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_dir,
        blendshape_deltas_path=None,  # blendshape_deltas_path,
        sampling_rate=sampling_rate,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=VOCARKitEvalDataset.collate_fn,
    )

    # Evaluate the data
    eval_metrics = evaluate(
        eval_dataloader=eval_dataloader,
        sampling_rate=sampling_rate,
        fps=fps,
        bc_threshold=bc_threshold,
    )

    print(eval_metrics)


if __name__ == "__main__":
    main()
