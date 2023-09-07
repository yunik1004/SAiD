"""Train the VAE for Blendshape Coefficients
"""
import argparse
from dataclasses import dataclass
import os
import pathlib
from typing import Optional
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from said.model.vae import BCVAE
from said.util.blendshape import load_blendshape_coeffs
from said.util.scheduler import frange_cycle_linear
from dataset.dataset_voca import BlendVOCAVAEDataset


@dataclass
class LossStepOutput:
    """
    Dataclass for the losses at each step
    """

    reconst: torch.FloatTensor  # (1,), Reconstruction loss
    regularize: torch.FloatTensor  # (1,), Regularization loss
    velocity: torch.FloatTensor  # (1,), Velocity loss


@dataclass
class LossEpochOutput:
    """
    Dataclass for the averaged losses at each epoch
    """

    total: float  # Averaged total loss
    reconst: float  # Averaged reconstruction loss
    regularize: float  # Averaged regularization loss
    velocity: float  # Averaged velocity loss
    lr: Optional[float] = None  # Last learning rate


def elbo_loss(
    said_vae: BCVAE,
    data: torch.FloatTensor,
    std: Optional[torch.FloatTensor],
    device: torch.device,
) -> LossStepOutput:
    """Compute the ELBO loss

    Parameters
    ----------
    said_vae : BCVAE
        BCVAE object
    data : torch.FloatTensor
        (Batch_size, sample_seq_len, x_dim), Input data
    std : Optional[torch.FloatTensor]
        (1, x_dim)
    device : torch.device
        GPU device

    Returns
    -------
    LossStepOutput
        Computed losses
    """
    blendshape_coeffs = data.to(device)
    output = said_vae(blendshape_coeffs)

    batch_size = blendshape_coeffs.shape[0]

    mean = output.mean
    log_var = output.log_var
    blendshape_coeffs_reconst = output.coeffs_reconst

    criterion_reconst = nn.MSELoss(reduction="sum")
    criterion_velocity = nn.MSELoss(reduction="sum")

    answer_reweight = blendshape_coeffs
    pred_reweight = blendshape_coeffs_reconst
    if std is not None:
        answer_reweight /= std.view(1, 1, -1)
        pred_reweight /= std.view(1, 1, -1)

    loss_reconst = 0.5 * criterion_reconst(answer_reweight, pred_reweight) / batch_size

    loss_kld = 0.5 * torch.mean(
        torch.sum(torch.pow(mean, 2) + torch.exp(log_var) - log_var - 1, dim=1)
    )

    answer_diff = answer_reweight[:, 1:, :] - answer_reweight[:, :-1, :]
    pred_diff = pred_reweight[:, 1:, :] - pred_reweight[:, :-1, :]

    loss_vel = 0.5 * criterion_velocity(pred_diff, answer_diff) / batch_size

    return LossStepOutput(
        reconst=loss_reconst,
        regularize=loss_kld,
        velocity=loss_vel,
    )


def train_epoch(
    said_vae: BCVAE,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    accelerator: Accelerator,
    beta: float,
    std: Optional[torch.FloatTensor],
    weight_vel: float,
    ema_vae: Optional[EMAModel] = None,
) -> LossEpochOutput:
    """Train the VAE one epoch.

    Parameters
    ----------
    said_vae : BCVAE
        BCVAE object
    train_dataloader : DataLoader
        Dataloader of the BlendVOCAVAEDataset
    optimizer : torch.optim.Optimizer
        Optimizer object
    lr_scheduler: torch.optim.lr_scheduler
        Learning rate scheduler object
    accelerator : Accelerator
        Accelerator object
    beta : float
        Loss weight
    std : Optional[torch.FloatTensor]
        (1, x_dim), Standard deviation of coefficients
    weight_vel: float
        Weight for the velocity loss
    ema_vae: Optional[EMAModel]
        EMA model of said_vae, by default None

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = accelerator.device
    if std is not None:
        std = std.to(device)

    said_vae.train()

    train_total_reconst_loss = 0
    train_total_reg_loss = 0
    train_total_vel_loss = 0
    train_total_loss = 0
    train_total_num = 0
    for data in train_dataloader:
        coeffs = data.blendshape_coeffs
        curr_batch_size = coeffs.shape[0]

        with accelerator.accumulate(said_vae):
            losses = elbo_loss(said_vae, coeffs, std, device)
            reconst_loss = losses.reconst
            reg_loss = losses.regularize
            velocity_loss = losses.velocity
            loss = reconst_loss + beta * reg_loss + weight_vel * velocity_loss

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(said_vae.parameters(), 1.0)

            optimizer.step()
            if ema_vae:
                ema_vae.step(said_vae.parameters())
            lr_scheduler.step()
            optimizer.zero_grad()

        train_total_num += curr_batch_size

        train_total_reconst_loss += reconst_loss.item() * curr_batch_size
        train_total_reg_loss += reg_loss.item() * curr_batch_size
        train_total_vel_loss += velocity_loss.item() * curr_batch_size
        train_total_loss += loss.item() * curr_batch_size

    train_avg_reconst_loss = train_total_reconst_loss / train_total_num
    train_avg_reg_loss = train_total_reg_loss / train_total_num
    train_avg_vel_loss = train_total_vel_loss / train_total_num
    train_avg_loss = train_total_loss / train_total_num

    return LossEpochOutput(
        total=train_avg_loss,
        reconst=train_avg_reconst_loss,
        regularize=train_avg_reg_loss,
        velocity=train_avg_vel_loss,
        lr=lr_scheduler.get_last_lr()[0],
    )


def validate_epoch(
    said_vae: BCVAE,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    beta: float,
    std: Optional[torch.FloatTensor],
    weight_vel: float,
    num_repeat: int = 1,
) -> LossEpochOutput:
    """Validate the VAE one epoch.

    Parameters
    ----------
    said_vae : BCVAE
        BCVAE object
    val_dataloader : DataLoader
        Dataloader of the BlendVOCAVAEDataset
    accelerator : Accelerator
        Accelerator object
    beta : float
        Loss weight
    std : Optional[torch.FloatTensor]
        (1, x_dim), Standard deviation of coefficients
    weight_vel: float
        Weight for the velocity loss
    num_repeat : int, optional
        Number of repetitions over whole validation dataset, by default 1

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = accelerator.device
    if std is not None:
        std = std.to(device)

    said_vae.eval()

    val_total_reconst_loss = 0
    val_total_reg_loss = 0
    val_total_vel_loss = 0
    val_total_loss = 0
    val_total_num = 0
    with torch.no_grad():
        for _ in range(num_repeat):
            for data in val_dataloader:
                coeffs = data.blendshape_coeffs
                curr_batch_size = coeffs.shape[0]

                losses = elbo_loss(said_vae, coeffs, std, device)
                reconst_loss = losses.reconst
                reg_loss = losses.regularize
                vel_loss = losses.velocity
                loss = reconst_loss + beta * reg_loss + weight_vel * vel_loss

                val_total_num += curr_batch_size

                val_total_reconst_loss += reconst_loss.item() * curr_batch_size
                val_total_reg_loss += reg_loss.item() * curr_batch_size
                val_total_vel_loss += vel_loss.item() * curr_batch_size
                val_total_loss += loss.item() * curr_batch_size

    val_avg_reconst_loss = val_total_reconst_loss / val_total_num
    val_avg_reg_loss = val_total_reg_loss / val_total_num
    val_avg_vel_loss = val_total_vel_loss / val_total_num
    val_avg_loss = val_total_loss / val_total_num

    return LossEpochOutput(
        total=val_avg_loss,
        reconst=val_avg_reconst_loss,
        regularize=val_avg_reg_loss,
        velocity=val_avg_vel_loss,
    )


def main():
    """Main function"""
    default_data_dir = pathlib.Path(__file__).parent.parent / "data"

    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using BlendVOCA dataset"
    )
    parser.add_argument(
        "--coeffs_dir",
        type=str,
        default="../BlendVOCA/blendshape_coeffs",
        help="Directory of the data",
    )
    parser.add_argument(
        "--coeffs_std_path",
        type=str,
        default="",  # (default_data_dir / "coeffs_std.csv").resolve(),  # "",
        help="Path of the coeffs std data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Directory of the outputs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size at training"
    )
    parser.add_argument(
        "--epochs", type=int, default=100000, help="The number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--beta", type=float, default=1, help="Beta for beta-VAE")
    parser.add_argument(
        "--beta_cycle",
        type=int,
        default=10,
        help="The number of cycles in beta schedule",
    )
    parser.add_argument(
        "--weight_vel",
        type=float,
        default=1.0,
        help="Weight for the velocity loss",
    )
    parser.add_argument(
        "--ema",
        type=bool,
        default=True,
        help="Use Exponential Moving Average of models weights",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="Ema decay rate",
    )
    parser.add_argument(
        "--val_period", type=int, default=500, help="Period of validating model"
    )
    parser.add_argument(
        "--val_repeat",
        type=int,
        default=10,
        help="Number of repetitions of the validation dataset",
    )
    parser.add_argument(
        "--save_period", type=int, default=500, help="Period of saving model"
    )
    args = parser.parse_args()

    coeffs_dir = args.coeffs_dir
    coeffs_std_path = args.coeffs_std_path
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    beta = args.beta
    beta_cycle = args.beta_cycle
    weight_vel = args.weight_vel
    ema = args.ema
    ema_decay = args.ema_decay
    val_period = args.val_period
    val_repeat = args.val_repeat
    save_period = args.save_period

    coeffs_std = (
        None if coeffs_std_path == "" else load_blendshape_coeffs(coeffs_std_path)
    )

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD-VAE")

    # Model
    said_vae = BCVAE()

    # Load data
    train_dataset = BlendVOCAVAEDataset(
        blendshape_coeffs_dir=coeffs_dir,
        dataset_type="train",
    )
    val_dataset = BlendVOCAVAEDataset(
        blendshape_coeffs_dir=coeffs_dir,
        dataset_type="val",
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
        collate_fn=BlendVOCAVAEDataset.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=BlendVOCAVAEDataset.collate_fn,
    )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(params=said_vae.parameters(), lr=learning_rate)

    num_training_steps = len(train_dataloader) * epochs

    lr_scheduler = get_scheduler(
        name="constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=0.1 * num_training_steps,
        num_training_steps=num_training_steps,
    )

    beta_schedules = frange_cycle_linear(n_iter=epochs, stop=beta, n_cycle=beta_cycle)

    # Prepare the acceleration using accelerator
    (
        said_vae,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
    ) = accelerator.prepare(
        said_vae, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    # Prepare the EMA model
    ema_vae = EMAModel(said_vae.parameters(), decay=ema_decay) if ema else None

    # Set the progress bar
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Epochs")

    for epoch in range(1, epochs + 1):
        # KL weight
        beta_epoch = beta_schedules[epoch - 1]

        # Train the model
        train_losses = train_epoch(
            said_vae=said_vae,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            beta=beta_epoch,
            std=coeffs_std,
            weight_vel=weight_vel,
            ema_vae=ema_vae,
        )

        # Log
        logs = {
            "Train/Total": train_losses.total,
            "Train/Reconst": train_losses.reconst,
            "Train/Regular": train_losses.regularize,
            "Train/Velocity": train_losses.velocity,
            "Train/Beta": beta_epoch,
            "Train/Learning Rate": train_losses.lr,
        }

        accelerator.wait_for_everyone()

        # Validate the model
        if epoch % val_period == 0:
            if ema:
                ema_vae.store(said_vae.parameters())
                ema_vae.copy_to(said_vae.parameters())

            val_losses = validate_epoch(
                said_vae=said_vae,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
                beta=beta_epoch,
                std=coeffs_std,
                weight_vel=weight_vel,
                num_repeat=val_repeat,
            )
            # Append the log
            logs["Val/Total"] = val_losses.total
            logs["Val/Reconst"] = val_losses.reconst
            logs["Val/Regular"] = val_losses.regularize
            logs["Val/Velocity"] = val_losses.velocity

            if ema:
                ema_vae.restore(said_vae.parameters())

        # Print logs
        if accelerator.sync_gradients:
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

        if accelerator.is_main_process:
            accelerator.log(logs, step=epoch)

        accelerator.wait_for_everyone()

        # Save the model
        if epoch % save_period == 0 and accelerator.is_main_process:
            if ema:
                ema_vae.store(said_vae.parameters())
                ema_vae.copy_to(said_vae.parameters())

            accelerator.save(
                accelerator.unwrap_model(said_vae).state_dict(),
                os.path.join(output_dir, f"{epoch}.pth"),
            )

            if ema:
                ema_vae.restore(said_vae.parameters())

    accelerator.end_training()


if __name__ == "__main__":
    main()
