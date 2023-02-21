"""Train the SAID_UNet1D model
"""
import argparse
import os
from typing import Any, Dict
from accelerate import Accelerator
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model
from said.model.diffusion import SAID, SAID_CDiT, SAID_UNet1D
from dataset import VOCARKitTrainDataset, VOCARKitValDataset


def random_noise_loss(
    said_model: SAID,
    data: Dict[str, Any],
    device: torch.device,
    mdm_like: bool = False,
) -> torch.FloatTensor:
    """Compute the MSE loss with randomized noises

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    data : Dict[str, Any]
        Output of the SAID.collate_fn
    device : torch.device
        GPU device
    mdm_like: bool
        Whether predict the signal itself or just a noise, by default False

    Returns
    -------
    torch.FloatTensor
        MSE loss between added noise and predicted noise
    """
    waveform = data["waveform"]
    blendshape_coeffs = data["blendshape_coeffs"].to(device)
    coeff_latents = said_model.encode_samples(
        blendshape_coeffs * said_model.latent_scale
    )

    curr_batch_size = len(waveform)

    waveform_processed = said_model.process_audio(waveform).to(device)
    random_timesteps = said_model.get_random_timesteps(curr_batch_size).to(device)

    audio_embedding = said_model.get_audio_embedding(waveform_processed)
    noise_dict = said_model.add_noise(coeff_latents, random_timesteps)
    noisy_latents = noise_dict["noisy_samples"]
    noise = noise_dict["noise"]

    pred = said_model(noisy_latents, random_timesteps, audio_embedding)

    criterion = nn.MSELoss()
    loss = criterion(coeff_latents if mdm_like else noise, pred)

    return loss


def train_epoch(
    said_model: SAID,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    mdm_like: bool = False,
) -> float:
    """Train the SAiD model one epoch.

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    train_dataloader : DataLoader
        Dataloader of the VOCARKitTrainDataset
    optimizer : torch.optim.Optimizer
        Optimizer object
    accelerator : Accelerator
        Accelerator object
    mdm_like: bool
        Whether predict the signal itself or just a noise, by default False

    Returns
    -------
    float
        Average loss
    """
    device = accelerator.device

    said_model.train()

    train_total_loss = 0
    train_total_num = 0
    for data in train_dataloader:
        optimizer.zero_grad()

        curr_batch_size = len(data["waveform"])
        loss = random_noise_loss(said_model, data, device, mdm_like)

        accelerator.backward(loss)
        optimizer.step()

        train_total_loss += loss.item() * curr_batch_size
        train_total_num += curr_batch_size

    train_avg_loss = train_total_loss / train_total_num

    return train_avg_loss


def validate_epoch(
    said_model: SAID,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    mdm_like: bool = False,
    num_repeat: int = 1,
) -> float:
    """Train the SAiD model one epoch.

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    val_dataloader : DataLoader
        Dataloader of the VOCARKitValDataset
    accelerator : Accelerator
        Accelerator object
    mdm_like: bool
        Whether predict the signal itself or just a noise, by default False
    num_repeat : int, optional
        Number of the repetition, by default 1

    Returns
    -------
    float
        Average loss
    """
    device = accelerator.device

    said_model.eval()

    val_total_loss = 0
    val_total_num = 0
    with torch.no_grad():
        for _ in range(num_repeat):
            for data in val_dataloader:
                curr_batch_size = len(data["waveform"])
                loss = random_noise_loss(said_model, data, device, mdm_like)

                val_total_loss += loss.item() * curr_batch_size
                val_total_num += curr_batch_size

    val_avg_loss = val_total_loss / val_total_num

    return val_avg_loss


def main():
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using VOCA-ARKit dataset"
    )
    """
    parser.add_argument(
        "--vae_weights_path",
        type=str,
        default="../output-vae/5000.pth",
        help="Path of the weights of VAE",
    )
    """
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../VOCA_ARKit",
        help="Directory of the data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Directory of the outputs",
    )
    parser.add_argument(
        "--mdm_like",
        type=bool,
        default=False,
        help="Whether predict the signal itself or just a noise",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=120,
        help="Window size of the blendshape coefficients sequence at training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size at training"
    )
    parser.add_argument(
        "--epochs", type=int, default=10000, help="The number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--uncond_prob",
        type=float,
        default=0.1,
        help="Unconditional probability of waveform (for classifier-free guidance)",
    )
    parser.add_argument(
        "--val_period", type=int, default=50, help="Period of validating model"
    )
    parser.add_argument(
        "--val_repeat", type=int, default=10, help="Number of repetition of val dataset"
    )
    parser.add_argument(
        "--save_period", type=int, default=200, help="Period of saving model"
    )
    args = parser.parse_args()

    # vae_weights_path = args.vae_weights_path

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_audio_dir = os.path.join(train_dir, "audio")
    train_blendshape_coeffs_dir = os.path.join(train_dir, "blendshape_coeffs")
    val_audio_dir = os.path.join(val_dir, "audio")
    val_blendshape_coeffs_dir = os.path.join(val_dir, "blendshape_coeffs")

    output_dir = args.output_dir
    mdm_like = args.mdm_like
    window_size = args.window_size
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    uncond_prob = args.uncond_prob
    val_period = args.val_period
    val_repeat = args.val_repeat
    save_period = args.save_period

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", logging_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD")

    # Load model with pretrained audio encoder, VAE
    # said_model = SAID_CDiT()
    said_model = SAID_UNet1D()
    said_model.audio_encoder = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    """
    said_model.vae.load_state_dict(
        torch.load(vae_weights_path, map_location=accelerator.device)
    )
    """

    # Load data
    train_dataset = VOCARKitTrainDataset(
        train_audio_dir,
        train_blendshape_coeffs_dir,
        said_model.sampling_rate,
        window_size,
        uncond_prob=uncond_prob,
    )
    val_dataset = VOCARKitTrainDataset(
        val_audio_dir,
        val_blendshape_coeffs_dir,
        said_model.sampling_rate,
        window_size,
        uncond_prob=uncond_prob,
    )
    """
    val_dataset = VOCARKitValDataset(
        val_audio_dir,
        val_blendshape_coeffs_dir,
        said_model.sampling_rate,
    )
    """

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=VOCARKitTrainDataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=VOCARKitValDataset.collate_fn,
    )

    # Initialize the optimzier - freeze audio encoder, VAE
    for p in said_model.audio_encoder.parameters():
        p.requires_grad = False

    """
    for p in said_model.vae.parameters():
        p.requires_grad = False
    """

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, said_model.parameters()),
        lr=learning_rate,
    )

    # Prepare the acceleration using accelerator
    said_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        said_model, optimizer, train_dataloader, val_dataloader
    )

    # Set the progress bar
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Epochs")

    for epoch in range(1, epochs + 1):
        # Train the model
        train_avg_loss = train_epoch(
            said_model=said_model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            accelerator=accelerator,
            mdm_like=mdm_like,
        )

        # Log
        logs = {"Train/Loss": train_avg_loss}

        # Validate the model
        if epoch % val_period == 0:
            val_avg_loss = validate_epoch(
                said_model=said_model,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
                mdm_like=mdm_like,
                num_repeat=val_repeat,
            )
            # Append the log
            logs["Validation/Loss"] = val_avg_loss

        # Print logs
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


if __name__ == "__main__":
    main()
