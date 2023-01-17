import argparse
from diffusers import DDIMScheduler
import torch
from said.model.diffusion import SAID_Wav2Vec2
from said.util.audio import load_audio


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using VOCA-ARKit dataset"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../output-prev/3000.pth",
        help="Path of the weights of SAiD model",
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default="../VOCA_ARKit/test/audio/FaceTalk_170913_03279_TA_sentence01.wav",
        help="Path of the audio file",
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
        res = said_model.inference(waveform_processed, None, 200)


if __name__ == "__main__":
    main()
