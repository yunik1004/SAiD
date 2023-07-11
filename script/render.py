"""Render the animation
Reference 1: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
Reference 2: https://github.com/Doubiiu/CodeTalker/blob/main/main/render.py
"""
import argparse
import os
import pathlib
from typing import List
import cv2
from moviepy import editor as mpy
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender
from tqdm import trange
import trimesh
from said.util.mesh import create_mesh, load_mesh
from said.util.parser import parse_list
from said.util.blendshape import load_blendshape_coeffs


def render_mesh(
    mesh: trimesh.Trimesh,
    t_center: np.ndarray,
    rot: np.ndarray = np.zeros(3),
    z_offset: float = 0,
) -> np.ndarray:
    """Render the mesh

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh object
    t_center: np.ndarray
        (3,), Position of the center
    rot: np.ndarray
        (3,), Rotation angle, by default np.zeros(3)
    z_offset: float
        Z offset of the camera, by default 0

    Returns
    -------
    np.ndarray
        (800, 800, 3), Rendered image
    """
    camera_params = {
        "c": np.array([400, 400]),
        "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
        "f": np.array([4754.97941935 / 2, 4754.97941935 / 2]),
    }

    frustum = {"near": 0.01, "far": 3.0, "height": 800, "width": 800}

    new_vertices = cv2.Rodrigues(rot)[0].dot((mesh.vertices - t_center).T).T + t_center
    mesh_copy = create_mesh(new_vertices, mesh.faces)

    intensity = 2.0
    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode="BLEND",
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8,
    )

    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[0, 0, 0])
    camera = pyrender.IntrinsicsCamera(
        fx=camera_params["f"][0],
        fy=camera_params["f"][1],
        cx=camera_params["c"][0],
        cy=camera_params["c"][1],
        znear=frustum["near"],
        zfar=frustum["far"],
    )

    render_mesh = pyrender.Mesh.from_trimesh(
        mesh_copy, material=primitive_material, smooth=True
    )
    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3, 3]
    light_color = np.array([1.0, 1.0, 1.0])
    light = pyrender.PointLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3, 3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    # Disable the following codes due to the bug related to the light (maybe pyrender's)
    # light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    # scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(
            viewport_width=frustum["width"], viewport_height=frustum["height"]
        )
        color, _ = r.render(scene, flags=flags)
        r.delete()
    except:
        print("pyrender: Failed rendering frame")
        color = np.zeros((frustum["height"], frustum["width"], 3), dtype="uint8")

    return color[..., ::-1].astype(np.uint8)


def render_blendshape_coefficients(
    neutral_mesh: trimesh.Trimesh,
    blendshapes_matrix: np.ndarray,
    blendshape_coeffs: np.ndarray,
) -> List[np.ndarray]:
    """Render the mesh

    Parameters
    ----------
    neutral_mesh: trimesh.Trimesh
        Mesh of the neutral object
    blendshapes_matrix: np.ndarray
        (3|V|, num_blendshapes), [b1 | b2 | ... | b_N] blendshape mesh's vertices vectors
    blendshape_coeffs: np.ndarray
        (T_b, num_classes), Blendshape coefficients

    mesh : trimesh.Trimesh
        Mesh object
    z_offset: float
        Z offset of the camera, by default 0

    Returns
    -------
    List[np.ndarray]
        Rendered images
    """
    neutral_vector = neutral_mesh.vertices.reshape((-1, 1))
    faces = neutral_mesh.faces
    blendshapes_matrix_delta = blendshapes_matrix - neutral_vector

    motion_vector_sequence = (
        blendshape_coeffs @ blendshapes_matrix_delta.T + neutral_vector.T
    )
    seq_len = motion_vector_sequence.shape[0]
    num_vertices = motion_vector_sequence.shape[1] // 3

    motion_vertex_sequence = motion_vector_sequence.reshape(seq_len, num_vertices, 3)

    center = np.mean(neutral_mesh.vertices, axis=0)

    rendered_imgs = []
    for sdx in trange(seq_len):
        vertices = motion_vertex_sequence[sdx]
        mesh = create_mesh(vertices, faces)
        rendered_img = render_mesh(mesh, center)
        rendered_imgs.append(rendered_img)

    return rendered_imgs


def main() -> None:
    """Main function"""
    default_data_dir = pathlib.Path(__file__).parent.parent / "data"

    # Arguments
    parser = argparse.ArgumentParser(description="Render the animation")
    parser.add_argument(
        "--neutral_path",
        type=str,
        default="../VOCA_ARKit/templates_head/FaceTalk_170731_00024_TA.obj",
        help="Path of the neutral mesh",
    )
    parser.add_argument(
        "--blendshapes_dir",
        type=str,
        default="../VOCA_ARKit/blendshapes_head/FaceTalk_170731_00024_TA",
        help="Directory of the blendshape meshes",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="../VOCA_ARKit/audio/FaceTalk_170731_00024_TA/sentence01.wav",
        help="Path of the audio file",
    )
    parser.add_argument(
        "--blendshape_coeffs_path",
        type=str,
        default="../VOCA_ARKit/blendshape_coeffs/FaceTalk_170731_00024_TA/sentence01.csv",
        help="Path of the blendshape coefficient sequence",
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
    fps = args.fps
    output_path = args.output_path
    save_images = args.save_images
    output_images_dir = args.output_images_dir

    blendshape_name_list = parse_list(blendshape_list_path, str)

    neutral_mesh = load_mesh(neutral_path)
    blendshape_vectors = []
    for bl_name in blendshape_name_list:
        bl_path = os.path.join(blendshapes_dir, f"{bl_name}.obj")
        bl_mesh = load_mesh(bl_path)
        blendshape_vectors.append(bl_mesh.vertices.reshape((-1, 1)))

    blendshapes_matrix = np.concatenate(blendshape_vectors, axis=1)
    blendshape_coeffs = load_blendshape_coeffs(blendshape_coeffs_path).numpy()

    # Render images
    rendered_imgs = render_blendshape_coefficients(
        neutral_mesh=neutral_mesh,
        blendshapes_matrix=blendshapes_matrix,
        blendshape_coeffs=blendshape_coeffs,
    )

    # Render video
    clip = mpy.ImageSequenceClip(rendered_imgs, fps=fps)
    clip = clip.set_audio(mpy.AudioFileClip(audio_path))
    clip.write_videofile(output_path, fps=fps)

    # Save rendered images
    if save_images:
        for idx in range(len(rendered_imgs)):
            img = rendered_imgs[idx]
            cv2.imwrite(os.path.join(output_images_dir, f"{idx}.png"), img)


if __name__ == "__main__":
    main()
