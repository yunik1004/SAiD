"""Inference reconstructed blendshape coefficients using VAE
"""
import argparse
import torch
from said.model.vae import BCVAE
from said.util.blendshape import load_blendshape_coeffs, save_blendshape_coeffs
from dataset import VOCARKIT_CLASSES


def main():
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Reconstruct the blendshape coefficients using VAE"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../output/3000.pth",
        help="Path of the weights of VAE",
    )
    parser.add_argument(
        "--blendshape_coeffs_path",
        type=str,
        default="../VOCA_ARKit/test/blendshape_coeffs/FaceTalk_170913_03279_TA_sentence01.csv",
        help="Path of the weights of VAE",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../out.csv",
        help="Path of the output blendshape coefficients file (csv format)",
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
    device = args.device

    # Load model
    said_vae = BCVAE()
    said_vae.load_state_dict(torch.load(weights_path, map_location=device))
    said_vae.to(device)
    said_vae.eval()

    # Load data
    blendshape_coeffs = load_blendshape_coeffs(blendshape_coeffs_path).unsqueeze(0)
    blendshape_coeffs.to(device)

    # Inference
    with torch.no_grad():
        output = said_vae(blendshape_coeffs)

    blendshape_coeffs_reconst = output["reconstruction"]

    save_blendshape_coeffs(
        coeffs=blendshape_coeffs_reconst[0].cpu().numpy(),
        classes=VOCARKIT_CLASSES,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
