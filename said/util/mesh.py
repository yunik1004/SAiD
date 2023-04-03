"""Define the utility functions related to the mesh
"""
from typing import Dict, List
import numpy as np
import trimesh


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load the mesh

    Parameters
    ----------
    filepath : str
        Path of the mesh file

    Returns
    -------
    trimesh.Trimesh
        Mesh object
    """
    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
    return mesh


def get_submesh(
    vertices: np.ndarray, faces: np.ndarray, subindices: List[int]
) -> Dict[str, np.ndarray]:
    """Get the submesh

    Parameters
    ----------
    vertices : np.ndarray
        (|V|, 3), Vertices of the mesh
    faces : np.ndarray
        (|F|, 3), Faces of the mesh
    subindices : List[int]
        Length of |V'|, indices of the submesh

    Returns
    -------
    Dict[str, np.ndarray]
        {
            "vertices": (|V'|, 3), Vertices of the submesh
            "faces":  (|F'|, 3), Faces of the submesh
        }
    """
    sub_vertices = vertices[subindices]

    sub_faces_list = []
    for face in faces:
        try:
            v0 = subindices.index(face[0])
            v1 = subindices.index(face[1])
            v2 = subindices.index(face[2])
            sub_faces_list.append([v0, v1, v2])
        except:
            pass
    sub_faces = np.array(sub_faces_list)

    out = {
        "vertices": sub_vertices,
        "faces": sub_faces,
    }

    return out


def create_mesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    """Create the trimesh

    Parameters
    ----------
    vertices : np.ndarray
        (|V|, 3), Vertices of the mesh
    faces : np.ndarray
        (|F|, 3), Faces of the mesh

    Returns
    -------
    trimesh.Trimesh
        Trimesh object
    """
    mesh = trimesh.Trimesh(vertices, faces)
    return mesh


def save_mesh(mesh: trimesh.Trimesh, out_path: str) -> None:
    """Save the mesh

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Trimesh object
    out_path : str
        Path of the output file
    """
    mesh.export(out_path)
