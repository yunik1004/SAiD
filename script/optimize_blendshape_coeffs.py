"""Generate the Pseudo-GT blendshape coefficients
"""
import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import numpy as np
import os
import pathlib
from tqdm import tqdm
from said.util.blendshape import save_blendshape_coeffs
from said.util.parser import parse_list
from said.optimize.blendshape_coeffs import OptimizationProblemFull
from dataset.dataset_voca import VOCARKitDataset, VOCARKitPseudoGTOptDataset


def main():
    """Main function"""
    default_data_dir = pathlib.Path(__file__).parent.parent / "data"

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
        default=(default_data_dir / "ARKit_blendshapes.txt").resolve(),
        help="List of the blendshapes",
    )
    parser.add_argument(
        "--head_idx_path",
        type=str,
        default=(default_data_dir / "FLAME_head_idx.txt").resolve(),
        help="List of the head indices",
    )
    parser.add_argument(
        "--blendshapes_coeffs_out_dir",
        type=str,
        default="../output_coeffs",
        help="Directory of the output coefficients",
    )
    parser.add_argument(
        "--num_repeat",
        type=int,
        default=20,
        help="Number of repetitions of the coefficients generation per each sequence",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=3,
        help="Number of processes for the multi-processing",
    )
    args = parser.parse_args()

    neutrals_dir = args.neutrals_dir
    blendshapes_dir = args.blendshapes_dir
    mesh_seqs_dir = args.mesh_seqs_dir

    blendshape_list_path = args.blendshape_list_path
    head_idx_path = args.head_idx_path

    blendshapes_coeffs_out_dir = args.blendshapes_coeffs_out_dir

    num_repeat = args.num_repeat
    num_process = args.num_process

    def coeff_out_path(
        person_id: str, seq_id: int, repeat_number: int, exists_ok: bool = False
    ) -> str:
        """Generate the output path of the coefficients.
        If you want to change the output file name, then change this function

        Parameters
        ----------
        person_id : str
            Person id
        seq_id : int
            Sequence id
        repeat_number: int
            Repetition ordinal
        exists_ok : bool, optional
            If false, raise error when the file already exists, by default False

        Returns
        -------
        str
            Output path of the coefficients
        """
        out_dir = os.path.join(blendshapes_coeffs_out_dir, person_id)
        try:
            os.makedirs(out_dir)
        except OSError:
            if not exists_ok:
                raise "Directory already exists"

        out_path = os.path.join(out_dir, f"sentence{seq_id:02}-{repeat_number}.csv")

        return out_path

    # Parse blendshape name
    blendshape_name_list = parse_list(blendshape_list_path, str)

    # Parse head indices
    head_idx_list = parse_list(head_idx_path, int)

    dataset = VOCARKitPseudoGTOptDataset(
        neutrals_dir, blendshapes_dir, mesh_seqs_dir, blendshape_name_list
    )

    person_id_list = (
        VOCARKitDataset.person_ids_train
        + VOCARKitDataset.person_ids_val
        + VOCARKitDataset.person_ids_test
    )
    seq_id_list = VOCARKitDataset.sentence_ids

    # Multi-processing
    pool = ProcessPoolExecutor(max_workers=num_process)

    for person_id in tqdm(person_id_list):
        bl_out = dataset.get_blendshapes(person_id)

        neutral_mesh = bl_out.neutral
        blendshapes_meshes_dict = bl_out.blendshapes

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
            mesh_seq_list = dataset.get_mesh_seq(person_id, seq_id)
            if len(mesh_seq_list) == 0:
                continue

            mesh_seq_vertices_list = [mesh.vertices for mesh in mesh_seq_list]
            mesh_seq_vertices_vector_list = [
                vertices[head_idx_list].reshape((-1, 1))
                for vertices in mesh_seq_vertices_list
            ]

            # Solve optimization problem
            procs = []
            for _ in range(num_repeat):
                init_vals = np.random.uniform(
                    size=(len(mesh_seq_vertices_vector_list), opt_prob.num_blendshapes)
                )
                procs.append(
                    pool.submit(
                        opt_prob.optimize, mesh_seq_vertices_vector_list, init_vals
                    )
                )

            # Save the coefficients
            for rdx, p in enumerate(as_completed(procs)):
                w_soln = p.result()

                out_path = coeff_out_path(person_id, seq_id, rdx, sdx > 0 or rdx > 0)
                save_blendshape_coeffs(
                    w_soln,
                    blendshape_name_list,
                    out_path,
                )


if __name__ == "__main__":
    main()
