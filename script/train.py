"""Train the SAID_UNet1D model
"""
import argparse
import os
from typing import Any, Dict, Optional
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from said.model.diffusion import SAID, SAID_CDiT, SAID_UNet1D
from said.model.wav2vec2 import ModifiedWav2Vec2Model
from dataset.dataset_voca import VOCARKitTrainDataset, VOCARKitValDataset


def random_noise_loss(
    said_model: SAID,
    data: Dict[str, Any],
    device: torch.device,
    mdm_like: bool = False,
) -> Dict[str, torch.FloatTensor]:
    """Compute the loss with randomized noises

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    data : Dict[str, Any]
        Output of the VOCARKitDataset.collate_fn
    device : torch.device
        GPU device
    mdm_like: bool
        Whether predict the signal itself or just a noise, by default False

    Returns
    -------
    Dict[str, torch.FloatTensor]
        {
            "loss_epsilon": MAE loss between added noise and predicted noise
            "loss_reconst": MSE loss between original vertices and predicted vertices
        }
    """
    waveform = data["waveform"]
    blendshape_coeffs = data["blendshape_coeffs"].to(device)
    blendshape_delta = data["blendshape_delta"].to(device)
    cond = data["cond"].to(device)

    coeff_latents = said_model.encode_samples(
        blendshape_coeffs * said_model.latent_scale
    )

    curr_batch_size = len(waveform)
    window_size = blendshape_coeffs.shape[1]

    waveform_processed = said_model.process_audio(waveform).to(device)
    random_timesteps = said_model.get_random_timesteps(curr_batch_size).to(device)

    audio_embedding = said_model.get_audio_embedding(waveform_processed, window_size)
    audio_embedding_cond = audio_embedding * cond.view(-1, 1, 1)

    noise_dict = said_model.add_noise(coeff_latents, random_timesteps)
    noisy_latents = noise_dict["noisy_samples"]
    noise = noise_dict["noise"]

    pred = said_model(noisy_latents, random_timesteps, audio_embedding_cond)

    criterion_epsilon = nn.L1Loss()
    loss_epsilon = criterion_epsilon(coeff_latents if mdm_like else noise, pred)

    losses = {
        "loss_epsilon": loss_epsilon,
    }

    criterion_reconst = nn.L1Loss()

    if mdm_like:
        # TODO: Not implemented & tested
        losses["loss_reconst"] = torch.tensor(0, device=device).float()
    else:
        """
        pred_latents = said_model.pred_original_sample(
            noisy_latents, pred, random_timesteps
        )
        latents_diff = pred_latents - coeff_latents
        """
        latents_diff = noise - pred
        vertices_diff = torch.einsum("bijk,bli->bljk", blendshape_delta, latents_diff)

        loss_reconst = criterion_reconst(vertices_diff, torch.zeros_like(vertices_diff))

        losses["loss_reconst"] = loss_reconst

    return losses


def train_epoch(
    said_model: SAID,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    mdm_like: bool = False,
    ema_model: Optional[EMAModel] = None,
    lambda_reconst: float = 1e2,
) -> Dict[str, float]:
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
    ema_model: Optional[EMAModel]
        EMA model of said_model, by default None
    lambda_reconst: float
        Loss weight of the reconstruction loss, by default 1e2

    Returns
    -------
    Dict[str, float]
        Average losses
        {
            "loss": Average loss to be optimized
            "loss_epsilon": Average epsilon loss
            "loss_reconst": Average reconstruction loss
        }
    """
    device = accelerator.device

    said_model.train()

    train_total_losses = {
        "loss": 0,
        "loss_epsilon": 0,
        "loss_reconst": 0,
    }
    train_total_num = 0
    for data in train_dataloader:
        curr_batch_size = len(data["waveform"])
        losses = random_noise_loss(said_model, data, device, mdm_like)

        loss_epsilon = losses["loss_epsilon"]
        loss_reconst = losses["loss_reconst"]

        loss = loss_epsilon + lambda_reconst * loss_reconst

        accelerator.backward(loss)
        optimizer.step()
        if ema_model:
            ema_model.step(said_model.parameters())
        optimizer.zero_grad()

        train_total_losses["loss"] += loss.item() * curr_batch_size
        train_total_losses["loss_epsilon"] += loss_epsilon.item() * curr_batch_size
        train_total_losses["loss_reconst"] += loss_reconst.item() * curr_batch_size

        train_total_num += curr_batch_size

    train_avg_losses = {k: v / train_total_num for k, v in train_total_losses.items()}

    return train_avg_losses


def validate_epoch(
    said_model: SAID,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    mdm_like: bool = False,
    num_repeat: int = 1,
    lambda_reconst: float = 1e2,
) -> Dict[str, float]:
    """Validate the SAiD model one epoch.

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
    lambda_reconst: float
        Loss weight of the reconstruction loss, by default 1e2

    Returns
    -------
    Dict[str, float]
        Average loss
        {
            "loss": Average loss to be optimized
            "loss_epsilon": Average epsilon loss
            "loss_reconst": Average reconstruction loss
        }
    """
    device = accelerator.device

    said_model.eval()

    val_total_losses = {
        "loss": 0,
        "loss_epsilon": 0,
        "loss_reconst": 0,
    }
    val_total_num = 0
    with torch.no_grad():
        for _ in range(num_repeat):
            for data in val_dataloader:
                curr_batch_size = len(data["waveform"])
                losses = random_noise_loss(said_model, data, device, mdm_like)

                loss_epsilon = losses["loss_epsilon"]
                loss_reconst = losses["loss_reconst"]

                loss = loss_epsilon + lambda_reconst * loss_reconst

                val_total_losses["loss"] += loss.item() * curr_batch_size
                val_total_losses["loss_epsilon"] += (
                    loss_epsilon.item() * curr_batch_size
                )
                val_total_losses["loss_reconst"] += (
                    loss_reconst.item() * curr_batch_size
                )

                val_total_num += curr_batch_size

    val_avg_losses = {k: v / val_total_num for k, v in val_total_losses.items()}

    return val_avg_losses


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
        "--audio_dir",
        type=str,
        default="../VOCA_ARKit/audio",
        help="Directory of the audio data",
    )
    parser.add_argument(
        "--coeffs_dir",
        type=str,
        default="../VOCA_ARKit/blendshape_coeffs",
        help="Directory of the blendshape coefficients data",
    )
    parser.add_argument(
        "--blendshape_deltas_path",
        type=str,
        default="../VOCA_ARKit/blendshape_deltas.pickle",
        help="Path of the blendshape deltas",
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
        "--epochs", type=int, default=20000, help="The number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--uncond_prob",
        type=float,
        default=0.25,
        help="Unconditional probability of waveform (for classifier-free guidance)",
    )
    parser.add_argument(
        "--ema",
        type=bool,
        default=True,
        help="Use Exponential Moving Average of models weights",
    )
    parser.add_argument(
        "--val_period", type=int, default=200, help="Period of validating model"
    )
    parser.add_argument(
        "--val_repeat", type=int, default=50, help="Number of repetition of val dataset"
    )
    parser.add_argument(
        "--save_period", type=int, default=200, help="Period of saving model"
    )
    args = parser.parse_args()

    # vae_weights_path = args.vae_weights_path

    audio_dir = args.audio_dir
    coeffs_dir = args.coeffs_dir
    blendshape_deltas_path = args.blendshape_deltas_path

    output_dir = args.output_dir
    mdm_like = args.mdm_like
    window_size = args.window_size
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    uncond_prob = args.uncond_prob
    ema = args.ema
    val_period = args.val_period
    val_repeat = args.val_repeat
    save_period = args.save_period

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD")

    # Load model with pretrained audio encoder, VAE
    # said_model = SAID_CDiT()
    said_model = SAID_UNet1D()
    said_model.audio_encoder = ModifiedWav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )
    """
    said_model.vae.load_state_dict(
        torch.load(vae_weights_path, map_location=accelerator.device)
    )
    """

    # Load data
    train_dataset = VOCARKitTrainDataset(
        audio_dir,
        coeffs_dir,
        blendshape_deltas_path,
        said_model.sampling_rate,
        window_size,
        uncond_prob=uncond_prob,
    )
    val_dataset = VOCARKitValDataset(
        audio_dir,
        coeffs_dir,
        blendshape_deltas_path,
        said_model.sampling_rate,
        uncond_prob=uncond_prob,
    )

    train_sampler = RandomSampler(
        train_dataset,
        replacement=True,
        num_samples=len(train_dataset),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
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
        weight_decay=1e-4,
    )

    # Prepare the acceleration using accelerator
    said_model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        said_model, optimizer, train_dataloader, val_dataloader
    )

    # Prepare the EMA model
    ema_model = EMAModel(said_model.parameters(), decay=0.99) if ema else None

    # Set the progress bar
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Epochs")

    for epoch in range(1, epochs + 1):
        # Train the model
        train_avg_losses = train_epoch(
            said_model=said_model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            accelerator=accelerator,
            mdm_like=mdm_like,
            ema_model=ema_model,
        )

        # Log
        logs = {
            "Train/Total Loss": train_avg_losses["loss"],
            "Train/Epsilon Loss": train_avg_losses["loss_epsilon"],
            "Train/Reconst Loss": train_avg_losses["loss_reconst"],
        }

        accelerator.wait_for_everyone()

        # EMA
        if ema and accelerator.is_main_process:
            ema_model.copy_to(said_model.parameters())

        # Validate the model
        if epoch % val_period == 0:
            val_avg_losses = validate_epoch(
                said_model=said_model,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
                mdm_like=mdm_like,
                num_repeat=val_repeat,
            )
            # Append the log
            logs["Validation/Total Loss"] = val_avg_losses["loss"]
            logs["Validation/Epsilon Loss"] = val_avg_losses["loss_epsilon"]
            logs["Validation/Reconst Loss"] = val_avg_losses["loss_reconst"]

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
