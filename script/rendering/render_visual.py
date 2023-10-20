"""
Render the visual outputs
Reference 1: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
Reference 2: https://github.com/Doubiiu/CodeTalker/blob/main/main/render.py
"""
from queue import Queue
from typing import List, Optional
import cv2
from matplotlib.pyplot import get_cmap
import numpy as np
from tqdm import trange
import trimesh
import pyrender
from said.util.mesh import create_mesh


class RendererObject:
    """
    Renderer wrapper
    """

    def __init__(self, z_offset: float = 0) -> None:
        """
        Constructor of the RendererObject

        z_offset: float
            Z offset of the camera, by default 0
        """
        camera_params = {
            "c": np.array([400, 400]),
            "k": np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            "f": np.array([4754.97941935 / 2, 4754.97941935 / 2]),
        }

        self.frustum = {"near": 0.01, "far": 3.0, "height": 800, "width": 800}

        intensity = 2.0
        self.primitive_material = pyrender.material.MetallicRoughnessMaterial(
            alphaMode="BLEND",
            baseColorFactor=[0.3, 0.3, 0.3, 1.0],
            metallicFactor=0.8,
            roughnessFactor=0.8,
        )

        self.scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=[0, 0, 0])
        camera = pyrender.IntrinsicsCamera(
            fx=camera_params["f"][0],
            fy=camera_params["f"][1],
            cx=camera_params["c"][0],
            cy=camera_params["c"][1],
            znear=self.frustum["near"],
            zfar=self.frustum["far"],
        )

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 1.0 - z_offset])
        self.scene.add(
            camera, pose=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
        )

        angle = np.pi / 6.0
        pos = camera_pose[:3, 3]
        light_color = np.array([1.0, 1.0, 1.0])
        light = pyrender.PointLight(color=light_color, intensity=intensity)

        light_pose = np.eye(4)
        light_pose[:3, 3] = pos
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        light_pose[:3, 3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
        self.scene.add(light, pose=light_pose.copy())

        # Disable the following codes due to the bug related to the light (maybe pyrender's)
        # light_pose[:3, 3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
        # self.scene.add(light, pose=light_pose.copy())

        self.r = pyrender.OffscreenRenderer(
            viewport_width=self.frustum["width"], viewport_height=self.frustum["height"]
        )

    def __del__(self):
        self.r.delete()

    def render(
        self,
        mesh: trimesh.Trimesh,
        t_center: np.ndarray,
        rot: np.ndarray = np.zeros(3),
        render_vertex_color: bool = False,
    ) -> np.ndarray:
        """
        mesh : trimesh.Trimesh
            Mesh object
        t_center: np.ndarray
            (3,), Position of the center
        rot: np.ndarray
            (3,), Rotation angle, by default np.zeros(3)
        render_vertex_color: bool
            Whether rendering vertex colors, by default False

        Returns
        -------
        np.ndarray
            (800, 800, 3), Rendered images
        """
        new_vertices = (
            cv2.Rodrigues(rot)[0].dot((mesh.vertices - t_center).T).T + t_center
        )
        mesh_copy = create_mesh(new_vertices, mesh.faces)
        mesh_copy.visual.vertex_colors = mesh.visual.vertex_colors

        render_material = None if render_vertex_color else self.primitive_material
        render_mesh = pyrender.Mesh.from_trimesh(
            mesh_copy, material=render_material, smooth=True
        )
        node_mesh = self.scene.add(render_mesh, pose=np.eye(4))

        flags = pyrender.RenderFlags.SKIP_CULL_FACES
        try:
            color, _ = self.r.render(self.scene, flags=flags)
        except:
            print("pyrender: Failed rendering frame")
            color = np.zeros(
                (self.frustum["height"], self.frustum["width"], 3), dtype="uint8"
            )

        self.scene.remove_node(node_mesh)

        return color[..., ::-1].astype(np.uint8)


def render_blendshape_coefficients(
    renderer: RendererObject,
    neutral_mesh: trimesh.Trimesh,
    blendshapes_matrix: np.ndarray,
    blendshape_coeffs: np.ndarray,
    target_blendshape_coeffs: Optional[np.ndarray] = None,
    color_map: str = "viridis",
    max_diff: float = 0.001,
) -> List[np.ndarray]:
    """Render the mesh from the blendshape coefficient sequence

    Parameters
    ----------
    neutral_mesh: trimesh.Trimesh
        Mesh of the neutral object
    blendshapes_matrix: np.ndarray
        (3|V|, num_blendshapes), [b1 | b2 | ... | b_N] blendshape mesh's vertices vectors
    blendshape_coeffs: np.ndarray
        (T_b, num_classes), Blendshape coefficients
    target_blendshape_coeffs: Optional[np.ndarray]
        (T_b, num_classes), Target blendshape coefficients to compute the vertex differences, by default None

    Returns
    -------
    List[np.ndarray]
        (800, 800, 3), Rendered images
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

    visualize_difference = target_blendshape_coeffs is not None

    if visualize_difference:
        difference = (
            (target_blendshape_coeffs - blendshape_coeffs) @ blendshapes_matrix_delta.T
        ).reshape(seq_len, num_vertices, 3)
        difference_mag = np.sqrt(np.sum(difference**2, axis=2))

        cmap = get_cmap(color_map)
        diff_mag_flatten = np.clip(difference_mag.reshape(-1), 0, max_diff) / max_diff
        vertex_colors = cmap(diff_mag_flatten).reshape(seq_len, num_vertices, 4)

        """
        vertex_colors = trimesh.visual.interpolate(
            difference_mag.reshape(-1), color_map=color_map
        ).reshape(seq_len, num_vertices, 4)
        """

    rendered_imgs = []
    for sdx, vseq in enumerate(motion_vertex_sequence):
        mesh = create_mesh(vseq, faces)
        if visualize_difference:
            mesh.visual.vertex_colors = vertex_colors[sdx]
        img = renderer.render(mesh, center, render_vertex_color=visualize_difference)
        rendered_imgs.append(img)

    return rendered_imgs
