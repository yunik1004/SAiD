"""Inference reconstructed blendshape coefficients using VAE
"""
import argparse
import pathlib
import torch
from said.model.vae import BCVAE
from said.util.blendshape import (
    load_blendshape_coeffs,
    save_blendshape_coeffs,
    save_blendshape_coeffs_image,
)
from dataset.dataset_voca import BlendVOCADataset


def main():
    """Main function"""
    default_model_dir = pathlib.Path(__file__).parent.parent / "model"

    # Arguments
    parser = argparse.ArgumentParser(
        description="Reconstruct the blendshape coefficients using VAE"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default=(default_model_dir / "vae.pth").resolve(),
        help="Path of the weights of VAE",
    )
    parser.add_argument(
        "--blendshape_coeffs_path",
        type=str,
        default="../BlendVOCA/blendshape_coeffs/FaceTalk_170731_00024_TA/sentence01.csv",
        help="Path of the input blendshape coefficients file (csv format)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../out.csv",
        help="Path of the output blendshape coefficients file (csv format)",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="../out.png",
        help="Path of the image of the output blendshape coefficients",
    )
    parser.add_argument(
        "--save_image",
        type=bool,
        default=False,
        help="Save the output blendshape coefficients as an image",
    )
    parser.add_argument(
        "--use_noise",
        type=bool,
        default=True,
        help="Use the noise when reconstructing the coefficients",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU/CPU device",
    )
    args = parser.parse_args()

    weights_path = args.weights_path
    blendshape_coeffs_path = args.blendshape_coeffs_path
    output_path = args.output_path
    output_image_path = args.output_image_path
    save_image = args.save_image
    use_noise = args.use_noise
    device = args.device

    # Load model
    said_vae = BCVAE()
    said_vae.load_state_dict(torch.load(weights_path, map_location=device))
    said_vae.to(device)
    said_vae.eval()

    # Load data
    blendshape_coeffs = (
        load_blendshape_coeffs(blendshape_coeffs_path)[: said_vae.seq_len]
        .unsqueeze(0)
        .to(device)
    )

    # Inference
    with torch.no_grad():
        output = said_vae(blendshape_coeffs, use_noise)

    blendshape_coeffs_reconst = output.coeffs_reconst

    result = blendshape_coeffs_reconst[0].cpu().numpy()

    save_blendshape_coeffs(
        coeffs=result,
        classes=BlendVOCADataset.default_blendshape_classes,
        output_path=output_path,
    )

    if save_image:
        save_blendshape_coeffs_image(result, output_image_path)


if __name__ == "__main__":
    main()
