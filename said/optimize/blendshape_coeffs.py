"""Define the optimization problem to parse blendshape coefficients
"""
from typing import List, Optional
import numpy as np
from qpsolvers import solve_qp
from scipy import linalg as la
from scipy import sparse as sp


class OptimizationProblemSingle:
    """Autoregressive optimization of pseudo-gt coefficients"""

    def __init__(
        self,
        neutral_vector: np.ndarray,
        blendshapes_matrix: np.ndarray,
    ) -> None:
        """Constructor of OptimizationProblemSingle

        Parameters
        ----------
        neutral_vector : np.ndarray
            (3|V|, 1), neutral mesh's vertices vector
        blendshapes_matrix : np.ndarray
            (3|V|, num_blendshapes), [b1 | b2 | ... | b_N] blendshape mesh's vertices vectors
        """
        self.neutral_vector = neutral_vector

        self.num_blendshapes = blendshapes_matrix.shape[1]

        # B_delta
        self.blendshapes_matrix_delta = blendshapes_matrix - self.neutral_vector

        self.P = self.blendshapes_matrix_delta.T @ self.blendshapes_matrix_delta

        self.lbw = np.zeros(self.num_blendshapes)
        self.ubw = np.ones(self.num_blendshapes)

    def optimize(
        self, vertices_vector: np.ndarray, init_vals: Optional[np.ndarray]
    ) -> np.ndarray:
        """Solve the optimization problem

        Parameters
        ----------
        vertices_vector : np.ndarray
            (3|V|, 1), Target mesh's vertices vector to be optimized
        init_vals : Optional[np.ndarray]
            (num_blendshapes,), Initial point of the optimization

        Returns
        -------
        np.ndarray
            (num_blendshapes,), optimization solution
        """
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
    """Full optimization of pseudo-gt coefficients"""

    def __init__(
        self,
        neutral_vector: np.ndarray,
        blendshapes_matrix: np.ndarray,
    ) -> None:
        """Constructor of OptimizationProblemFull

        Parameters
        ----------
        neutral_vector : np.ndarray
            (3|V|, 1), neutral mesh's vertices vector
        blendshapes_matrix : np.ndarray
            (3|V|, num_blendshapes), [b1 | b2 | ... | b_N] blendshape mesh's vertices vectors
        """
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
        self,
        vertices_vector_list: List[np.ndarray],
        init_vals: Optional[np.ndarray] = None,
        delta: float = 0.1,
    ) -> np.ndarray:
        """Solve the optimization problem

        Parameters
        ----------
        vertices_vector_list : List[np.ndarray]
            (3|V|, 1), List of the target mesh sequence's vertices vectors to be optimized
        init_vals: Optional[np.ndarray]
            (seq_len, num_blendshapes), initial value of the optimization
        delta : float, optional
            Bound of the |w_{t} - w_{t+1}|, by default 0.1

        Returns
        -------
        np.ndarray
            (seq_len, num_blendshapes), optimization solution
        """
        seq_len = len(vertices_vector_list)

        # Compute P
        p = la.block_diag(*[self.btb for _ in range(seq_len)])

        # Compute q
        q = np.vstack(
            [
                self.blendshapes_matrix_delta.T @ (self.neutral_vector - vvector)
                for vvector in vertices_vector_list
            ]
        ).reshape(-1)

        # Compute G
        g = self.compute_g(seq_len)

        # Set h
        h = np.full(g.shape[0], delta)

        # Uppler/lower bound
        lbw = np.zeros(self.num_blendshapes * seq_len)
        ubw = np.ones(self.num_blendshapes * seq_len)

        # Solve the problem
        w_sol = solve_qp(
            P=p,
            q=q,
            G=g,
            h=h,
            lb=lbw,
            ub=ubw,
            solver="cvxopt",
            initvals=None if init_vals is None else init_vals.reshape(-1),
        )
        w_sol = np.clip(w_sol, lbw, ubw)

        w_vectors_matrix = w_sol.reshape(seq_len, self.num_blendshapes)

        return w_vectors_matrix

    def compute_g(self, seq_len: int) -> sp.csc_matrix:
        """Compute G efficiently using sparse matrix

        Parameters
        ----------
        seq_len : int
            Length of the target mesh sequence

        Returns
        -------
        sp.csc_matrix
            ((seq_len - 1) * num_blendshapes, seq_len * num_blendshapes), computed G
        """
        diag_g = sp.block_diag(
            [self.dipole_eye for _ in range(seq_len - 1)], format="csc"
        )

        pos_g = sp.block_diag((diag_g, self.g_offset), format="csc")
        neg_g = sp.block_diag((self.g_offset, diag_g), format="csc")

        g = pos_g - neg_g
        return g
