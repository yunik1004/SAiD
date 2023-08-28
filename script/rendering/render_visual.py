"""
Render the visual outputs
Reference 1: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
Reference 2: https://github.com/Doubiiu/CodeTalker/blob/main/main/render.py
"""
from typing import List
import cv2
import numpy as np
from tqdm import trange
import trimesh
import pyrender
from said.util.mesh import create_mesh


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
