"""Train the VAE for Blendshape Coefficients
"""
import argparse
from dataclasses import dataclass
import os
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from said.model.vae import BCVAE
from dataset.dataset_voca import VOCARKitVAEDataset


@dataclass
class LossStepOutput:
    """
    Dataclass for the losses at each step
    """

    reconst: torch.FloatTensor  # (1,), Reconstruction loss
    regularize: torch.FloatTensor  # (1,), Regularization loss


@dataclass
class LossEpochOutput:
    """
    Dataclass for the averaged losses at each epoch
    """

    total: float  # Averaged total loss
    reconst: float  # Averaged reconstruction loss
    regularize: float  # Averaged regularization loss


def elbo_loss(
    said_vae: BCVAE, data: torch.FloatTensor, device: torch.device
) -> LossStepOutput:
    """Compute the ELBO loss

    Parameters
    ----------
    said_vae : BCVAE
        BCVAE object
    data : torch.FloatTensor
        (Batch_size, sample_seq_len, x_dim), Input data
    device : torch.device
        GPU device

    Returns
    -------
    LossStepOutput
        Computed losses
    """
    blendshape_coeffs = data.to(device)
    output = said_vae(blendshape_coeffs)

    mean = output.mean
    log_var = output.log_var
    blendshape_coeffs_reconst = output.coeffs_reconst

    l1_func = torch.nn.L1Loss()

    reconst_loss = l1_func(blendshape_coeffs_reconst, blendshape_coeffs)
    kld_loss = 0.5 * torch.mean(torch.pow(mean, 2) + torch.exp(log_var) - log_var - 1)

    return LossStepOutput(reconst=reconst_loss, regularize=kld_loss)


def train_epoch(
    said_vae: BCVAE,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    beta: float,
) -> LossEpochOutput:
    """Train the VAE one epoch.

    Parameters
    ----------
    said_vae : BCVAE
        BCVAE object
    train_dataloader : DataLoader
        Dataloader of the VOCARKitVAEDataset
    optimizer : torch.optim.Optimizer
        Optimizer object
    accelerator : Accelerator
        Accelerator object
    beta : float, optional
        Loss weight

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = accelerator.device

    said_vae.train()

    train_total_reconst_loss = 0
    train_total_reg_loss = 0
    train_total_loss = 0
    train_total_num = 0
    for data in train_dataloader:
        optimizer.zero_grad()

        coeffs = data.blendshape_coeffs
        curr_batch_size = coeffs.shape[0]

        losses = elbo_loss(said_vae, coeffs, device)
        reconst_loss = losses.reconst
        reg_loss = losses.regularize
        loss = reconst_loss + beta * reg_loss

        accelerator.backward(loss)
        optimizer.step()

        train_total_num += curr_batch_size

        train_total_reconst_loss += reconst_loss.item() * curr_batch_size
        train_total_reg_loss += reg_loss.item() * curr_batch_size
        train_total_loss += loss.item() * curr_batch_size

    train_avg_reconst_loss = train_total_reconst_loss / train_total_num
    train_avg_reg_loss = train_total_reg_loss / train_total_num
    train_avg_loss = train_total_loss / train_total_num

    return LossEpochOutput(
        total=train_avg_loss,
        reconst=train_avg_reconst_loss,
        regularize=train_avg_reg_loss,
    )


def validate_epoch(
    said_vae: BCVAE,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    beta: float,
    num_repeat: int = 1,
) -> LossEpochOutput:
    """Validate the VAE one epoch.

    Parameters
    ----------
    said_vae : BCVAE
        BCVAE object
    val_dataloader : DataLoader
        Dataloader of the VOCARKitVAEDataset
    accelerator : Accelerator
        Accelerator object
    beta : float
        Loss weight
    num_repeat : int, optional
        Number of repetitions over whole validation dataset, by default 1

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = accelerator.device

    said_vae.eval()

    val_total_reconst_loss = 0
    val_total_reg_loss = 0
    val_total_loss = 0
    val_total_num = 0
    with torch.no_grad():
        for _ in range(num_repeat):
            for data in val_dataloader:
                coeffs = data.blendshape_coeffs
                curr_batch_size = coeffs.shape[0]

                losses = elbo_loss(said_vae, coeffs, device)
                reconst_loss = losses.reconst
                reg_loss = losses.regularize
                loss = reconst_loss + beta * reg_loss

                val_total_num += curr_batch_size

                val_total_reconst_loss += reconst_loss.item() * curr_batch_size
                val_total_reg_loss += reg_loss.item() * curr_batch_size
                val_total_loss += loss.item() * curr_batch_size

    val_avg_reconst_loss = val_total_reconst_loss / val_total_num
    val_avg_reg_loss = val_total_reg_loss / val_total_num
    val_avg_loss = val_total_loss / val_total_num

    return LossEpochOutput(
        total=val_avg_loss, reconst=val_avg_reconst_loss, regularize=val_avg_reg_loss
    )


def main():
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using VOCA-ARKit dataset"
    )
    parser.add_argument(
        "--coeffs_dir",
        type=str,
        default="../VOCA_ARKit/blendshape_coeffs",
        help="Directory of the data",
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
        "--epochs", type=int, default=20000, help="The number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--beta", type=float, default=1e-2, help="Loss weight")
    parser.add_argument(
        "--val_period", type=int, default=50, help="Period of validating model"
    )
    parser.add_argument(
        "--val_repeat",
        type=int,
        default=10,
        help="Number of repetitions of the validation dataset",
    )
    parser.add_argument(
        "--save_period", type=int, default=200, help="Period of saving model"
    )
    args = parser.parse_args()

    coeffs_dir = args.coeffs_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    beta = args.beta
    val_period = args.val_period
    val_repeat = args.val_repeat
    save_period = args.save_period

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD-VAE")

    # Model
    said_vae = BCVAE()

    # Load data
    train_dataset = VOCARKitVAEDataset(
        blendshape_coeffs_dir=coeffs_dir,
        dataset_type="train",
    )
    val_dataset = VOCARKitVAEDataset(
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
        collate_fn=VOCARKitVAEDataset.collate_fn,
    )
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=VOCARKitVAEDataset.collate_fn,
    )

    # Initialize the optimizer
    optimizer = torch.optim.Adam(params=said_vae.parameters(), lr=learning_rate)

    # Prepare the acceleration using accelerator
    said_vae, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        said_vae, optimizer, train_dataloader, val_dataloader
    )

    # Set the progress bar
    progress_bar = tqdm(
        range(1, epochs + 1), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Epochs")

    for epoch in range(1, epochs + 1):
        # KL weight
        beta_epoch = beta

        # Train the model
        train_losses = train_epoch(
            said_vae=said_vae,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            accelerator=accelerator,
            beta=beta_epoch,
        )

        # Log
        logs = {
            "Train/Total": train_losses.total,
            "Train/Reconst": train_losses.reconst,
            "Train/Regular": train_losses.regularize,
            "Train/Beta": beta_epoch,
        }

        # Validate the model
        if epoch % val_period == 0:
            val_losses = validate_epoch(
                said_vae=said_vae,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
                beta=beta_epoch,
                num_repeat=val_repeat,
            )
            # Append the log
            logs["Val/Total"] = val_losses.total
            logs["Val/Reconst"] = val_losses.reconst
            logs["Val/Regular"] = val_losses.regularize

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
                accelerator.unwrap_model(said_vae).state_dict(),
                os.path.join(output_dir, f"{epoch}.pth"),
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
