"""Generate the inference results from the test data
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from said.model.diffusion import SAID_UNet1D
from said.util.audio import fit_audio_unet
from said.util.blendshape import save_blendshape_coeffs
from dataset.dataset_voca import VOCARKitDataset, VOCARKitTestDataset


def main() -> None:
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the output using VOCA-ARKit dataset"
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        default="../output/12000.pth",
        help="Path of the weights of SAiD model",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="../VOCA_ARKit/audio",
        help="Directory of the audio data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Directory of the outputs",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="Prediction type of the scheduler function, 'epsilon', 'sample', or 'v_prediction'",
    )
    parser.add_argument(
        "--num_steps", type=int, default=100, help="Number of inference steps"
    )
    parser.add_argument("--strength", type=float, default=1.0, help="How much to paint")
    parser.add_argument(
        "--guidance_scale", type=float, default=2.5, help="Guidance scale"
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
        default=1,
        help="Length of the blendshape coefficients sequence should be divided by this number",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU/CPU device",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=10,
        help="Number of repetitions in inference for each audio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Set the negative value if you don't want to control the randomness",
    )
    args = parser.parse_args()

    weights_path = args.weights_path
    audio_dir = args.audio_dir
    output_dir = args.output_dir
    prediction_type = args.prediction_type
    num_steps = args.num_steps
    strength = args.strength
    guidance_scale = args.guidance_scale
    eta = args.eta
    fps = args.fps
    divisor_unet = args.divisor_unet
    device = args.device

    num_repeats = args.num_repeats
    seed = args.seed

    # Set random seed
    if seed >= 0:
        torch.manual_seed(seed)

    # Load model
    said_model = SAID_UNet1D(prediction_type=prediction_type)
    said_model.load_state_dict(torch.load(weights_path, map_location=device))
    said_model.to(device)
    said_model.eval()

    # Load data
    test_dataset = VOCARKitTestDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=None,
        blendshape_deltas_path=None,
        sampling_rate=said_model.sampling_rate,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=VOCARKitTestDataset.collate_fn,
    )

    with torch.no_grad():
        for ddx, data in enumerate(tqdm(test_dataloader)):
            waveform = torch.from_numpy(data.waveform[0]).to(device)
            data_path = test_dataset.data_paths[ddx]

            pid = data_path.person_id
            audio_path = data_path.audio
            output_filename_base = os.path.splitext(os.path.basename(audio_path))[0]
            output_file_dir = os.path.join(output_dir, pid)
            if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)

            # Fit the size of waveform
            fit_output = fit_audio_unet(
                waveform, said_model.sampling_rate, fps, divisor_unet
            )
            waveform = fit_output.waveform
            window_len = fit_output.window_size

            # Process the waveform
            waveform_processed = said_model.process_audio(waveform).to(device)

            for rdx in range(num_repeats):
                # Inference
                output = said_model.inference(
                    waveform_processed=waveform_processed,
                    num_inference_steps=num_steps,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    eta=eta,
                    show_process=False,
                )

                result = output.result[0, :window_len].cpu().numpy()

                output_filename = f"{output_filename_base}-{rdx}.csv"
                output_path = os.path.join(output_file_dir, output_filename)

                save_blendshape_coeffs(
                    coeffs=result,
                    classes=VOCARKitDataset.default_blendshape_classes,
                    output_path=output_path,
                )


if __name__ == "__main__":
    main()
