"""Render the animation of dataset
"""
import argparse
import os
import pathlib

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
from moviepy import editor as mpy
import numpy as np
from tqdm import tqdm
from said.util.mesh import load_mesh
from said.util.parser import parse_list
from said.util.blendshape import load_blendshape_coeffs
from dataset.dataset_voca import VOCARKitEvalDataset
from rendering.render_visual import render_blendshape_coefficients_multiprocess


def main() -> None:
    """Main function"""
    default_data_dir = pathlib.Path(__file__).parent.parent / "data"

    # Arguments
    parser = argparse.ArgumentParser(description="Render the animation of the dataset")
    parser.add_argument(
        "--neutral_dir",
        type=str,
        default="../VOCA_ARKit/templates_head",
        help="Directory of the neutral mesh data",
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
        "--blendshapes_dir",
        type=str,
        default="../VOCA_ARKit/blendshapes_head",
        help="Directory of the blendshape meshes",
    )
    parser.add_argument(
        "--blendshape_list_path",
        type=str,
        default=(default_data_dir / "ARKit_blendshapes.txt").resolve(),
        help="List of the blendshapes",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS of the blendshape coefficients sequence",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=8,
        help="The number of processes",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../out_render",
        help="Saving directory of the output video files",
    )
    args = parser.parse_args()

    neutral_dir = args.neutral_dir
    audio_dir = args.audio_dir
    coeffs_dir = args.coeffs_dir
    blendshapes_dir = args.blendshapes_dir
    blendshape_list_path = args.blendshape_list_path
    fps = args.fps
    num_process = args.num_process
    output_dir = args.output_dir

    blendshape_name_list = parse_list(blendshape_list_path, str)

    # Load data
    eval_dataset = VOCARKitEvalDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_dir,
        blendshape_deltas_path=None,
        sampling_rate=-1,
        preload=False,
    )

    neutral_meshes = {}
    blendshapes_matrices = {}
    for pid in eval_dataset.person_ids_test:
        neutral_meshes[pid] = load_mesh(os.path.join(neutral_dir, f"{pid}.obj"))

        blendshape_vectors = []
        for bl_name in blendshape_name_list:
            bl_path = os.path.join(blendshapes_dir, pid, f"{bl_name}.obj")
            bl_mesh = load_mesh(bl_path)
            blendshape_vectors.append(bl_mesh.vertices.reshape((-1, 1)))

        blendshapes_matrices[pid] = np.concatenate(blendshape_vectors, axis=1)

    for data_path in tqdm(eval_dataset.data_paths):
        pid = data_path.person_id
        audio_path = data_path.audio
        blendshape_coeffs_path = data_path.blendshape_coeffs
        filename = pathlib.Path(blendshape_coeffs_path).stem
        output_parent = os.path.join(output_dir, pid)
        if not os.path.exists(output_parent):
            os.makedirs(output_parent)
        output_path = os.path.join(output_parent, f"{filename}.mp4")

        blendshape_coeffs = load_blendshape_coeffs(blendshape_coeffs_path).numpy()

        # Render images
        rendered_imgs = render_blendshape_coefficients_multiprocess(
            neutral_mesh=neutral_meshes[pid],
            blendshapes_matrix=blendshapes_matrices[pid],
            blendshape_coeffs=blendshape_coeffs,
            num_process=num_process,
        )

        # Render video
        audio_clip = mpy.AudioFileClip(audio_path)
        clip = mpy.ImageSequenceClip(rendered_imgs, fps=fps)
        clip = clip.set_audio(audio_clip)
        clip.write_videofile(output_path, fps=fps, logger=None, threads=4)

        audio_clip.close()
        clip.close()


if __name__ == "__main__":
    main()
