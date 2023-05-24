"""Inference using the auxiliary decoder
"""
import argparse
import torch
from said.model.aux_decoder import AuxDecoder
from said.model.diffusion import SAID_CDiT, SAID_UNet1D
from said.util.audio import load_audio
from said.util.blendshape import save_blendshape_coeffs, save_blendshape_coeffs_image
from dataset import VOCARKIT_CLASSES


def main():
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Inference the lipsync using the auxiliary decoder"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../output-aux/10000.pth",
        help="Path of the weights of auxiliary decoder",
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
        "--output_image_path",
        type=str,
        default="../out.png",
        help="Path of the image of the output blendshape coefficients",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="FPS of the blendshape coefficients sequence",
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
    output_image_path = args.output_image_path
    fps = args.fps
    device = args.device

    # Load model, and pretrained audio encoder
    aux_decoder = AuxDecoder()
    aux_decoder.load_state_dict(torch.load(weights_path, map_location=device))
    aux_decoder.to(device)
    aux_decoder.eval()

    said_model = SAID_UNet1D()
    said_model.to(device)
    said_model.eval()

    # Load data
    waveform = load_audio(audio_path, said_model.sampling_rate)

    # Fit the size of waveform
    waveform_len = waveform.shape[0]
    window_len = int(waveform_len / said_model.sampling_rate * fps)

    # Process the waveform
    waveform_processed = said_model.process_audio(waveform).to(device)

    # Inference
    with torch.no_grad():
        audio_embedding = said_model.get_audio_embedding(waveform_processed, window_len)
        output = aux_decoder(audio_embedding)

    result = output[0].cpu().numpy()

    save_blendshape_coeffs(
        coeffs=result,
        classes=VOCARKIT_CLASSES,
        output_path=output_path,
    )

    save_blendshape_coeffs_image(
        coeffs=result,
        output_path=output_image_path,
    )


if __name__ == "__main__":
    main()
