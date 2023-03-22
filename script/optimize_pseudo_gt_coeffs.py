"""Generate the Pseudo-GT blendshape coefficients
"""
import argparse
from typing import List, Optional
import numpy as np
import os
from qpsolvers import solve_qp
from tqdm import tqdm
from said.util.blendshape import save_blendshape_coeffs
from said.util.parser import parse_list
from dataset import VOCARKitPseudoGTOptDataset


class OptimizationProblem:
    def __init__(
        self,
        neutral_vector: np.ndarray,
        blendshapes_matrix: np.ndarray,
        landmark_idx_list: List[int],
        landmark_weight: float = 10.0,
    ) -> None:
        self.neutral_vector = neutral_vector

        self.threev, self.num_blendshapes = blendshapes_matrix.shape

        # B_delta
        self.blendshapes_matrix_delta = blendshapes_matrix - self.neutral_vector

        self.P = self.blendshapes_matrix_delta.T @ self.blendshapes_matrix_delta

        self.lbw = np.zeros(self.num_blendshapes)
        self.ubw = np.ones(self.num_blendshapes)

    def optimize(
        self, vertices_vector: np.ndarray, init_vals: Optional[np.ndarray]
    ) -> np.ndarray:
        q = (
            self.blendshapes_matrix_delta.T @ (self.neutral_vector - vertices_vector)
        ).reshape(-1)

        w_sol = solve_qp(
            self.P,
            q,
            lb=self.lbw,
            ub=self.ubw,
            solver="cvxopt",
            initvals=init_vals,
        )
        w_sol = np.clip(w_sol, self.lbw, self.ubw)

        return w_sol


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate the Pseudo-GT blendshape coefficients by solving the optimization problem"
    )
    parser.add_argument(
        "--neutrals_dir",
        type=str,
        default="../VOCA_ARKit/templates_head",
        help="Directory of the neutral meshes",
    )
    parser.add_argument(
        "--blendshapes_dir",
        type=str,
        default="../VOCA_ARKit/blendshapes_head",
        help="Directory of the blendshape meshes",
    )
    parser.add_argument(
        "--mesh_seqs_dir",
        type=str,
        default="../VOCA_ARKit/unposedcleaneddata",
        help="Directory of the mesh sequences",
    )
    parser.add_argument(
        "--blendshape_list_path",
        type=str,
        default="../VOCA_ARKit/blendshapes.txt",
        help="List of the blendshapes",
    )
    parser.add_argument(
        "--landmark_list_path",
        type=str,
        default="../VOCA_ARKit/landmarks.txt",
        help="List of the landmark indices",
    )
    parser.add_argument(
        "--head_idx_path",
        type=str,
        default="../VOCA_ARKit/flame_head_idx.txt",
        help="List of the head indices",
    )
    parser.add_argument(
        "--blendshapes_coeffs_out_dir",
        type=str,
        default="../output_coeffs_lmk",
        help="Directory of the output coefficients",
    )
    args = parser.parse_args()

    neutrals_dir = args.neutrals_dir
    blendshapes_dir = args.blendshapes_dir
    mesh_seqs_dir = args.mesh_seqs_dir

    blendshape_list_path = args.blendshape_list_path
    landmark_list_path = args.landmark_list_path
    head_idx_path = args.head_idx_path

    blendshapes_coeffs_out_dir = args.blendshapes_coeffs_out_dir

    def coeff_out_path(person_id: str, seq_id: int, exists_ok: bool = False) -> str:
        out_dir = os.path.join(blendshapes_coeffs_out_dir, person_id)
        try:
            os.makedirs(out_dir)
        except OSError:
            if not exists_ok:
                raise "Directory already exists"

        out_path = os.path.join(out_dir, f"{seq_id:02}.csv")

        return out_path

    # Parse blendshape name
    blendshape_name_list = parse_list(blendshape_list_path, str)

    # Parse landmark indices
    landmark_idx_list = parse_list(landmark_list_path, int)

    # Parse head indices
    head_idx_list = parse_list(head_idx_path, int)

    dataset = VOCARKitPseudoGTOptDataset(
        neutrals_dir, blendshapes_dir, mesh_seqs_dir, blendshape_name_list
    )

    person_id_list = dataset.get_person_id_list()
    seq_id_list = dataset.get_seq_id_list()

    for person_id in tqdm(person_id_list):
        bl_out = dataset.get_blendshapes(person_id)

        neutral_mesh = bl_out["neutral"]
        blendshapes_meshes_dict = bl_out["blendshapes"]

        neutral_vertices = neutral_mesh.vertices
        blendshapes_vertices_list = [
            blendshapes_meshes_dict[name].vertices for name in blendshape_name_list
        ]

        neutral_vector = neutral_vertices.reshape((-1, 1))
        blendshape_vectors = []
        for v in blendshapes_vertices_list:
            blendshape_vectors.append(v.reshape((-1, 1)))

        blendshapes_matrix = np.concatenate(blendshape_vectors, axis=1)

        # Define the optimization problem
        opt_prob = OptimizationProblem(
            neutral_vector, blendshapes_matrix, landmark_idx_list
        )

        for sdx, seq_id in enumerate(tqdm(seq_id_list, leave=False)):
            out_path = coeff_out_path(person_id, seq_id, sdx > 0)

            mesh_seq_list = dataset.get_mesh_seq(person_id, seq_id)
            mesh_seq_vertices_list = [mesh.vertices for mesh in mesh_seq_list]

            blendshape_weights_list = []

            # Solve Optimization problem
            for vdx, vertices in enumerate(mesh_seq_vertices_list):
                vertices_vector = vertices[head_idx_list].reshape((-1, 1))
                w_soln = opt_prob.optimize(
                    vertices_vector,
                    blendshape_weights_list[vdx - 1] if vdx > 0 else None,
                )
                blendshape_weights_list.append(w_soln)

            save_blendshape_coeffs(
                blendshape_weights_list,
                blendshape_name_list,
                out_path,
            )


if __name__ == "__main__":
    main()
