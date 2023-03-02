"""Train the auxiliary decoder
"""
import argparse
import os
from typing import Any, Dict, Optional
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from said.model.aux_decoder import AuxDecoder
from said.model.diffusion import SAID, SAID_CDiT, SAID_UNet1D
from said.model.wav2vec2 import ModifiedWav2Vec2Model
from dataset import VOCARKitTrainDataset, VOCARKitValDataset


def aux_loss(
    decoder: AuxDecoder,
    said_model: SAID,
    data: Dict[str, Any],
    device: torch.device,
) -> torch.FloatTensor:
    """Compute the auxiliary Loss

    Parameters
    ----------
    decoder : AuxDecoder
        AuxDecoder object
    said_model : SAID
        SAID model object
    data : Dict[str, Any]
        Output of the VOCARKitDataset.collate_fn
    device : torch.device
        GPU device

    Returns
    -------
    torch.FloatTensor
        MAE loss between coefficients and predicted coefficients
    """
    waveform = data["waveform"]
    blendshape_coeffs = data["blendshape_coeffs"].to(device)
    coeff_latents = said_model.encode_samples(
        blendshape_coeffs * said_model.latent_scale
    )

    curr_batch_size = len(waveform)
    window_size = blendshape_coeffs.shape[1]

    waveform_processed = said_model.process_audio(waveform).to(device)
    audio_embedding = said_model.get_audio_embedding(waveform_processed, window_size)

    pred = decoder(audio_embedding)

    criterion = nn.L1Loss()
    loss = criterion(coeff_latents, pred)

    return loss


def train_epoch(
    decoder: AuxDecoder,
    said_model: SAID,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    ema_model: Optional[EMAModel] = None,
) -> float:
    """Train the auxiliary decoder one epoch.

    Parameters
    ----------
    decoder : AuxDecoder
        AuxDecoder object
    said_model : SAID
        SAID model object
    train_dataloader : DataLoader
        Dataloader of the VOCARKitTrainDataset
    optimizer : torch.optim.Optimizer
        Optimizer object
    accelerator : Accelerator
        Accelerator object
    ema_model : Optional[EMAModel], optional
        EMA model of auxiliary decoder, by default None

    Returns
    -------
    float
        Average loss
    """
    device = accelerator.device

    decoder.train()

    train_total_loss = 0
    train_total_num = 0
    for data in train_dataloader:
        curr_batch_size = len(data["waveform"])
        loss = aux_loss(decoder, said_model, data, device)

        accelerator.backward(loss)
        optimizer.step()
        if ema_model:
            ema_model.step(decoder.parameters())
        optimizer.zero_grad()

        train_total_loss += loss.item() * curr_batch_size
        train_total_num += curr_batch_size

    train_avg_loss = train_total_loss / train_total_num

    return train_avg_loss


def validate_epoch(
    decoder: AuxDecoder,
    said_model: SAID,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    num_repeat: int = 1,
) -> float:
    """Validate the auxiliary decoder model one epoch.

    Parameters
    ----------
    decoder : AuxDecoder
        AuxDecoder object
    said_model : SAID
        SAID model object
    val_dataloader : DataLoader
        Dataloader of the VOCARKitValDataset
    accelerator : Accelerator
        Accelerator object
    num_repeat : int, optional
        Number of the repetition, by default 1

    Returns
    -------
    float
        Average loss
    """
    device = accelerator.device

    decoder.eval()

    val_total_loss = 0
    val_total_num = 0
    with torch.no_grad():
        for _ in range(num_repeat):
            for data in val_dataloader:
                curr_batch_size = len(data["waveform"])
                loss = aux_loss(decoder, said_model, data, device)

                val_total_loss += loss.item() * curr_batch_size
                val_total_num += curr_batch_size

    val_avg_loss = val_total_loss / val_total_num

    return val_avg_loss


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train the auxiliary decoder using VOCA-ARKit dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../VOCA_ARKit",
        help="Directory of the data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output-aux",
        help="Directory of the outputs",
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
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--ema",
        type=bool,
        default=True,
        help="Use Exponential Moving Average of models weights",
    )
    parser.add_argument(
        "--val_period", type=int, default=50, help="Period of validating model"
    )
    parser.add_argument(
        "--val_repeat", type=int, default=10, help="Number of repetition of val dataset"
    )
    parser.add_argument(
        "--save_period", type=int, default=50, help="Period of saving model"
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_audio_dir = os.path.join(train_dir, "audio")
    train_blendshape_coeffs_dir = os.path.join(train_dir, "blendshape_coeffs")
    val_audio_dir = os.path.join(val_dir, "audio")
    val_blendshape_coeffs_dir = os.path.join(val_dir, "blendshape_coeffs")

    output_dir = args.output_dir
    window_size = args.window_size
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    ema = args.ema
    val_period = args.val_period
    val_repeat = args.val_repeat
    save_period = args.save_period

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD-aux")

    # Load model, and pretrained audio encoder
    aux_decoder = AuxDecoder()

    said_model = SAID_UNet1D()
    said_model.audio_encoder = ModifiedWav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )

    # Load data
    train_dataset = VOCARKitTrainDataset(
        train_audio_dir,
        train_blendshape_coeffs_dir,
        said_model.sampling_rate,
        window_size,
        uncond_prob=0,
    )
    val_dataset = VOCARKitTrainDataset(
        val_audio_dir,
        val_blendshape_coeffs_dir,
        said_model.sampling_rate,
        window_size,
        uncond_prob=0,
    )

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

    # Initialize the optimzier - freeze audio encoder
    for p in said_model.parameters():
        p.requires_grad = False
    said_model.eval()

    optimizer = torch.optim.AdamW(
        params=aux_decoder.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    # Prepare the acceleration using accelerator
    (
        aux_decoder,
        said_model,
        optimizer,
        train_dataloader,
        val_dataloader,
    ) = accelerator.prepare(
        aux_decoder, said_model, optimizer, train_dataloader, val_dataloader
    )

    # Prepare the EMA model
    ema_model = EMAModel(aux_decoder.parameters(), decay=0.99) if ema else None

    # Set the progress bar
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Epochs")

    for epoch in range(1, epochs + 1):
        # Train the model
        train_avg_loss = train_epoch(
            decoder=aux_decoder,
            said_model=said_model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            accelerator=accelerator,
            ema_model=ema_model,
        )

        # Log
        logs = {"Train/Loss": train_avg_loss}

        accelerator.wait_for_everyone()

        # EMA
        if ema and accelerator.is_main_process:
            ema_model.copy_to(aux_decoder.parameters())

        # Validate the model
        if epoch % val_period == 0:
            val_avg_loss = validate_epoch(
                decoder=aux_decoder,
                said_model=said_model,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
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
                accelerator.unwrap_model(aux_decoder).state_dict(),
                os.path.join(output_dir, f"{epoch}.pth"),
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
