"""Train the SAiD_Wav2Vec2 model
"""
import argparse
import os
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model
from said.model.diffusion import SAID_Wav2Vec2
from dataset import VOCARKitTrainDataset

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description="Train the SAiD model")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="../data_train/audio",
        help="Directory of the audio data",
    )
    parser.add_argument(
        "--blendshape_coeffs_dir",
        type=str,
        default="../data_train/blendshape_coeffs",
        help="Directory of the blendshape coefficients data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Directory of the outputs",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=120,
        help="Window size of the blendshape coefficients sequence",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="The number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--save_period", type=int, default=100, help="Period of saving model"
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir
    blendshape_coeffs_dir = args.blendshape_coeffs_dir
    output_dir = args.output_dir
    window_size = args.window_size
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    save_period = args.save_period

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", logging_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD")
    device = accelerator.device

    # Load model with pretrained audio encoder
    said_model = SAID_Wav2Vec2()
    said_model.audio_encoder = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )

    # Load data
    train_dataset = VOCARKitTrainDataset(
        audio_dir, blendshape_coeffs_dir, said_model.sampling_rate, window_size
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=VOCARKitTrainDataset.collate_fn,
    )

    # Initialize the optimzier
    optimizer = torch.optim.Adam(said_model.parameters(), lr=learning_rate)

    # Prepare the acceleration using accelerator
    said_model, optimizer, train_dataloader = accelerator.prepare(
        said_model, optimizer, train_dataloader
    )

    # Set the progress bar
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Epochs")

    for epoch in range(1, epochs + 1):
        # Train the model
        said_model.train()
        total_loss = 0
        total_num = 0
        for data in train_dataloader:
            waveform = data["waveform"]
            blendshape_coeffs = data["blendshape_coeffs"].to(device)
            curr_batch_size = len(waveform)
            volume_coeffs = torch.numel(blendshape_coeffs)

            optimizer.zero_grad()

            waveform_processed = said_model.process_audio(waveform).to(device)
            random_timesteps = said_model.get_random_timesteps(curr_batch_size).to(
                device
            )

            audio_embedding = said_model.get_audio_embedding(waveform_processed)
            noisy_coeffs, noise = said_model.add_noise(
                blendshape_coeffs, random_timesteps
            )

            noise_pred = said_model(noisy_coeffs, random_timesteps, audio_embedding)

            loss = torch.norm(noise_pred - noise) / volume_coeffs

            accelerator.backward(loss)
            optimizer.step()

            total_loss += loss.item() * curr_batch_size
            total_num += curr_batch_size

        avg_loss = total_loss / total_num

        # Print logs
        logs = {"loss": avg_loss}

        if accelerator.sync_gradients:
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            accelerator.log(logs, step=epoch)

        accelerator.wait_for_everyone()

        # Save the model
        if epoch % save_period == 0 and accelerator.is_main_process:
            accelerator.save(
                accelerator.unwrap_model(said_model).state_dict(),
                os.path.join(output_dir, f"{epoch}.pth"),
            )

    accelerator.end_training()
