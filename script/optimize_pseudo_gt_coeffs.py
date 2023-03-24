"""Generate the Pseudo-GT blendshape coefficients
"""
import argparse
from typing import List, Optional
import numpy as np
import os
from qpsolvers import solve_qp
from scipy import linalg as la
from scipy import sparse as sp
from tqdm import tqdm
from said.util.blendshape import save_blendshape_coeffs
from said.util.parser import parse_list
from dataset import VOCARKitPseudoGTOptDataset


class OptimizationProblemSingle:
    def __init__(
        self,
        neutral_vector: np.ndarray,
        blendshapes_matrix: np.ndarray,
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


class OptimizationProblemFull:
    def __init__(
        self,
        neutral_vector: np.ndarray,
        blendshapes_matrix: np.ndarray,
    ) -> None:
        self.neutral_vector = neutral_vector
        self.num_blendshapes = blendshapes_matrix.shape[1]

        # B_delta
        self.blendshapes_matrix_delta = blendshapes_matrix - self.neutral_vector
        self.btb = self.blendshapes_matrix_delta.T @ self.blendshapes_matrix_delta

        # D
        eye = sp.identity(self.num_blendshapes, dtype="int", format="csc")
        self.dipole_eye = sp.bmat([[eye], [-eye]])

        self.g_offset = sp.csc_matrix((0, self.num_blendshapes), dtype="int")

    def optimize(
        self, vertices_vector_list: List[np.ndarray], delta: float = 0.1
    ) -> np.ndarray:
        t_len = len(vertices_vector_list)

        # Compute P
        p = la.block_diag(*[self.btb for _ in range(t_len)])

        # Compute q
        q = np.vstack(
            [
                self.blendshapes_matrix_delta.T @ (self.neutral_vector - vvector)
                for vvector in vertices_vector_list
            ]
        ).reshape(-1)

        # Compute G
        g = self.compute_g(t_len)

        # Set h
        h = np.full(g.shape[0], delta)

        # Uppler/lower bound
        lbw = np.zeros(self.num_blendshapes * t_len)
        ubw = np.ones(self.num_blendshapes * t_len)

        # Solve the problem
        w_sol = solve_qp(
            P=p,
            q=q,
            G=g,
            h=h,
            lb=lbw,
            ub=ubw,
            solver="cvxopt",
        )
        w_sol = np.clip(w_sol, lbw, ubw)

        w_vectors_matrix = w_sol.reshape(t_len, self.num_blendshapes)

        return w_vectors_matrix

    def compute_g(self, t_len: int) -> sp.csc_matrix:
        diag_g = sp.block_diag(
            [self.dipole_eye for _ in range(t_len - 1)], format="csc"
        )

        pos_g = sp.block_diag((diag_g, self.g_offset), format="csc")
        neg_g = sp.block_diag((self.g_offset, diag_g), format="csc")

        g = pos_g - neg_g
        return g


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
        opt_prob = OptimizationProblemFull(neutral_vector, blendshapes_matrix)

        for sdx, seq_id in enumerate(tqdm(seq_id_list, leave=False)):
            out_path = coeff_out_path(person_id, seq_id, sdx > 0)

            mesh_seq_list = dataset.get_mesh_seq(person_id, seq_id)
            if len(mesh_seq_list) == 0:
                continue

            mesh_seq_vertices_list = [mesh.vertices for mesh in mesh_seq_list]
            mesh_seq_vertices_vector_list = [
                vertices[head_idx_list].reshape((-1, 1))
                for vertices in mesh_seq_vertices_list
            ]

            # Solve Optimization problem
            w_soln = opt_prob.optimize(mesh_seq_vertices_vector_list)

            save_blendshape_coeffs(
                w_soln,
                blendshape_name_list,
                out_path,
            )


if __name__ == "__main__":
    main()
