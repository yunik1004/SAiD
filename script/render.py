"""Render the animation
"""
import argparse
import os
import pathlib

os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
from moviepy import editor as mpy
import numpy as np
from said.util.mesh import load_mesh
from said.util.parser import parse_list
from said.util.blendshape import load_blendshape_coeffs
from rendering.render_visual import RendererObject, render_blendshape_coefficients


def main() -> None:
    """Main function"""
    default_data_dir = pathlib.Path(__file__).parent.parent / "data"

    # Arguments
    parser = argparse.ArgumentParser(description="Render the animation")
    parser.add_argument(
        "--neutral_path",
        type=str,
        default="../BlendVOCA/templates_head/FaceTalk_170731_00024_TA.obj",
        help="Path of the neutral mesh",
    )
    parser.add_argument(
        "--blendshapes_dir",
        type=str,
        default="../BlendVOCA/blendshapes_head/FaceTalk_170731_00024_TA",
        help="Directory of the blendshape meshes",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="../BlendVOCA/audio/FaceTalk_170731_00024_TA/sentence01.wav",
        help="Path of the audio file",
    )
    parser.add_argument(
        "--blendshape_coeffs_path",
        type=str,
        default="../BlendVOCA/blendshape_coeffs/FaceTalk_170731_00024_TA/sentence01.csv",
        help="Path of the blendshape coefficient sequence",
    )
    parser.add_argument(
        "--blendshape_list_path",
        type=str,
        default=(default_data_dir / "ARKit_blendshapes.txt").resolve(),
        help="List of the blendshapes",
    )
    parser.add_argument(
        "--show_difference",
        type=bool,
        default=False,
        help="Show the vertex differences from the target blendshape coefficients as a heatmap",
    )
    parser.add_argument(
        "--target_diff_blendshape_coeffs_path",
        type=str,
        default="../BlendVOCA/blendshape_coeffs/FaceTalk_170731_00024_TA/sentence01.csv",
        help="Path of the target blendshape coefficient sequence to compute the vertex differences. Its length should be same as the source's.",
    )
    parser.add_argument(
        "--max_diff",
        type=float,
        default=0.001,
        help="Maximum threshold to visualize the vertex differences",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS of the blendshape coefficients sequence",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../out.mp4",
        help="Path of the output video file",
    )
    parser.add_argument(
        "--save_images",
        type=bool,
        default=False,
        help="Save the image for each frame",
    )
    parser.add_argument(
        "--output_images_dir",
        type=str,
        default="../out_imgs",
        help="Saving directory of the output image for each frame",
    )
    args = parser.parse_args()

    neutral_path = args.neutral_path
    blendshapes_dir = args.blendshapes_dir
    audio_path = args.audio_path
    blendshape_coeffs_path = args.blendshape_coeffs_path
    blendshape_list_path = args.blendshape_list_path
    show_difference = args.show_difference
    target_diff_blendshape_coeffs_path = args.target_diff_blendshape_coeffs_path
    max_diff = args.max_diff
    fps = args.fps
    output_path = args.output_path
    save_images = args.save_images
    output_images_dir = args.output_images_dir

    blendshape_name_list = parse_list(blendshape_list_path, str)

    # Create renderer
    renderer = RendererObject()

    neutral_mesh = load_mesh(neutral_path)
    blendshape_vectors = []
    for bl_name in blendshape_name_list:
        bl_path = os.path.join(blendshapes_dir, f"{bl_name}.obj")
        bl_mesh = load_mesh(bl_path)
        blendshape_vectors.append(bl_mesh.vertices.reshape((-1, 1)))

    blendshapes_matrix = np.concatenate(blendshape_vectors, axis=1)
    blendshape_coeffs = load_blendshape_coeffs(blendshape_coeffs_path).numpy()
    target_blendshape_coeffs = (
        load_blendshape_coeffs(target_diff_blendshape_coeffs_path).numpy()
        if show_difference
        else None
    )

    # Render images
    rendered_imgs = render_blendshape_coefficients(
        renderer=renderer,
        neutral_mesh=neutral_mesh,
        blendshapes_matrix=blendshapes_matrix,
        blendshape_coeffs=blendshape_coeffs,
        target_blendshape_coeffs=target_blendshape_coeffs,
        max_diff=max_diff,
    )

    # Render video
    audio_clip = mpy.AudioFileClip(audio_path)
    clip = mpy.ImageSequenceClip(rendered_imgs, fps=fps)
    clip = clip.set_audio(audio_clip)
    clip.write_videofile(output_path, fps=fps)

    audio_clip.close()
    clip.close()

    # Save rendered images
    if save_images:
        for idx in range(len(rendered_imgs)):
            img = rendered_imgs[idx]
            cv2.imwrite(os.path.join(output_images_dir, f"{idx}.png"), img)


if __name__ == "__main__":
    main()
