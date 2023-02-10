"""Inference using the SAID_UNet1D model
"""
import argparse
import math
from diffusers import DDIMScheduler
import torch
from said.model.diffusion import SAID_UNet1D
from said.util.audio import load_audio
from said.util.blendshape import save_blendshape_coeffs, save_blendshape_coeffs_image
from dataset import VOCARKIT_CLASSES


def main():
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Inference the lipsync using the SAiD model"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../output/3000.pth",
        help="Path of the weights of SAiD model",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="../VOCA_ARKit/test/audio/FaceTalk_170913_03279_TA_sentence01.wav",
        help="Path of the audio file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../out.csv",
        help="Path of the output blendshape coefficients file (csv format)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=0.0, help="Guidance scale"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0, help="Eta for DDIMScheduler, between [0, 1]"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS of the blendshape coefficients sequence",
    )
    parser.add_argument(
        "--divisor_unet",
        type=int,
        default=8,
        help="Length of the blendshape coefficients sequence should be divided by this number",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU/CPU device",
    )
    parser.add_argument(
        "--save_intermediate",
        type=bool,
        default=False,
        help="Save the intermediate outputs",
    )
    args = parser.parse_args()

    weights_path = args.weights_path
    audio_path = args.audio_path
    output_path = args.output_path
    num_steps = args.num_steps
    guidance_scale = args.guidance_scale
    eta = args.eta
    fps = args.fps
    divisor_unet = args.divisor_unet
    device = args.device
    save_intermediate = args.save_intermediate

    # Load model
    said_model = SAID_UNet1D(
        noise_scheduler=DDIMScheduler(beta_schedule="squaredcos_cap_v2")
    )
    said_model.load_state_dict(torch.load(weights_path, map_location=device))
    said_model.to(device)
    said_model.eval()

    # Load data
    waveform = load_audio(audio_path, said_model.sampling_rate)

    # Fit the size of waveform
    gcd = math.gcd(said_model.sampling_rate, fps)
    divisor_waveform = said_model.sampling_rate // gcd * divisor_unet
    divisor_window = fps // gcd * divisor_unet

    waveform_len = waveform.shape[0]
    window_len = int(waveform_len / said_model.sampling_rate * fps)
    waveform_len_fit = math.ceil(waveform_len / divisor_waveform) * divisor_waveform

    if waveform_len_fit > waveform_len:
        tmp = torch.zeros(waveform_len_fit)
        tmp[:waveform_len] = waveform[:]
        waveform = tmp

    # Process the waveform
    waveform_processed = said_model.process_audio(waveform).to(device)

    # Inference
    with torch.no_grad():
        output = said_model.inference(
            waveform_processed=waveform_processed,
            init_samples=None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            eta=eta,
            save_intermediate=save_intermediate,
        )

    result = output["Result"][0, :window_len].cpu().numpy()
    intermediate = output["Intermediate"]

    save_blendshape_coeffs(
        coeffs=result,
        classes=VOCARKIT_CLASSES,
        output_path=output_path,
    )

    """
    save_blendshape_coeffs_image(result, "../out.png")

    if save_intermediate:
        for i, interm in enumerate(intermediate):
            save_blendshape_coeffs_image(
                interm[0, :window_len].cpu().numpy(), f"../interm/{i}.png"
            )
    """


if __name__ == "__main__":
    main()
