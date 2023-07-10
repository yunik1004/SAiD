"""Preprocess the VOCA dataset to generate ARKit blendshapes
"""
import argparse
import os
import pathlib
from said.util.mesh import save_mesh
from dataset.dataset_voca import VOCARKitDataset


def main():
    """Main function"""
    default_data_dir = pathlib.Path(__file__).parent.parent / "data"

    # Arguments
    parser = argparse.ArgumentParser(
        description="Preprocess the VOCA dataset and generate ARKit blendshapes"
    )
    parser.add_argument(
        "--templates_dir",
        type=str,
        default="../VOCA_ARKit/templates",
        help="Directory of the template meshes",
    )
    parser.add_argument(
        "--blendshape_residuals_path",
        type=str,
        default=(default_data_dir / "blendshape_residuals.pickle").resolve(),
        help="Path of the blendshape residuals",
    )
    parser.add_argument(
        "--blendshapes_out_dir",
        type=str,
        default="../output_blendshapes",
        help="Directory of output blendshapes",
    )
    args = parser.parse_args()

    templates_dir = args.templates_dir
    blendshape_deltas_path = args.blendshape_residuals_path
    out_dir = args.blendshapes_out_dir

    templates_head_dir = os.path.join(out_dir, "templates_head")
    blendshapes_head_dir = os.path.join(out_dir, "blendshapes_head")

    # Generate the output directory
    os.makedirs(templates_head_dir)
    os.makedirs(blendshapes_head_dir)

    # Preprocess the blendshapes
    bls = VOCARKitDataset.preprocess_blendshapes(templates_dir, blendshape_deltas_path)

    # Save the blendshapes
    for pid, bases in bls.items():
        neutral_path = os.path.join(templates_head_dir, f"{pid}.obj")
        bl_dir = os.path.join(blendshapes_head_dir, f"{pid}")
        os.makedirs(bl_dir)

        neutral_mesh = bases.neutral
        bl_meshes = bases.blendshapes

        # Save neutral
        save_mesh(neutral_mesh, neutral_path)

        # Save blendshapes
        for bl_name, bl_mesh in bl_meshes.items():
            bl_path = os.path.join(bl_dir, f"{bl_name}.obj")
            save_mesh(bl_mesh, bl_path)


if __name__ == "__main__":
    main()
