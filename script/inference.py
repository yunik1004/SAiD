"""Inference using the SAiD_Wav2Vec2 model
"""
import argparse
from diffusers import DDIMScheduler
import pandas as pd
import torch
from said.model.diffusion import SAID_Wav2Vec2
from said.util.audio import load_audio
from dataset import VOCARKIT_CLASSES


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using VOCA-ARKit dataset"
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
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0, help="Eta for DDIMScheduler, between [0, 1]"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU/CPU device",
    )
    args = parser.parse_args()

    weights_path = args.weights_path
    audio_path = args.audio_path
    output_path = args.output_path
    num_steps = args.num_steps
    guidance_scale = args.guidance_scale
    eta = args.eta
    device = args.device

    # Load model
    said_model = SAID_Wav2Vec2(noise_scheduler=DDIMScheduler())
    said_model.load_state_dict(torch.load(weights_path, map_location=device))
    said_model.to(device)
    said_model.eval()

    # Load data
    waveform = load_audio(audio_path, said_model.sampling_rate)
    waveform_processed = said_model.process_audio(waveform).to(device)

    # Inference
    with torch.no_grad():
        output = said_model.inference(
            waveform_processed=waveform_processed,
            init_samples=None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            eta=eta,
        )

    pout = pd.DataFrame(output[0].cpu().numpy(), columns=VOCARKIT_CLASSES)
    pout.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
