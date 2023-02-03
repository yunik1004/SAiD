"""Train the VAE for Blendshape Coefficients
"""
import argparse
import os
from typing import Dict
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from said.model.vae import BCVAE
from dataset import VOCARKitVAEDataset


def elbo_loss(
    said_vae: BCVAE, data: torch.FloatTensor, device: torch.device
) -> Dict[str, torch.FloatTensor]:
    """Compute the ELBO loss

    Parameters
    ----------
    said_vae : BCVAE
        BCVAE object
    data : torch.FloatTensor
        Input data
    device : torch.device
        GPU device

    Returns
    -------
    Dict[str, torch.FloatTensor]
        {
            "reconstruction": (1,), Reconstruction loss
            "regularization": (1,), Regularization loss
        }
    """
    batch_size = data.shape[0]
    blendshape_coeffs = data.to(device)
    output = said_vae(blendshape_coeffs)

    mean = output["mean"]
    log_var = output["log_var"]
    blendshape_coeffs_reconst = output["reconstruction"]

    mse_func = torch.nn.MSELoss(reduction="sum")

    reconst_loss = mse_func(blendshape_coeffs_reconst, blendshape_coeffs)
    kld_loss = (
        torch.sum(torch.pow(mean, 2) + torch.exp(log_var) - log_var - 1) / batch_size
    )

    output = {
        "reconstruction": reconst_loss,
        "regularization": kld_loss,
    }
    return output


def train_epoch(
    said_vae: BCVAE,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    beta: float,
) -> Dict[str, float]:
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
    Dict[str, float]
        {
            "reconstruction": Reconstruction loss
            "regularization": Regularization loss
            "total": Total loss
        }
    """
    device = accelerator.device

    said_vae.train()

    train_total_reconst_loss = 0
    train_total_reg_loss = 0
    train_total_loss = 0
    train_total_num = 0
    for data in train_dataloader:
        optimizer.zero_grad()

        curr_batch_size = data.shape[0]

        losses = elbo_loss(said_vae, data, device)
        reconst_loss = losses["reconstruction"]
        reg_loss = losses["regularization"]
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

    output = {
        "reconstruction": train_avg_reconst_loss,
        "regularization": train_avg_reg_loss,
        "total": train_avg_loss,
    }
    return output


def validate_epoch(
    said_vae: BCVAE,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    beta: float,
) -> Dict[str, float]:
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

    Returns
    -------
    Dict[str, float]
        {
            "reconstruction": Reconstruction loss
            "regularization": Regularization loss
            "total": Total loss
        }
    """
    device = accelerator.device

    said_vae.eval()

    val_total_reconst_loss = 0
    val_total_reg_loss = 0
    val_total_loss = 0
    val_total_num = 0
    with torch.no_grad():
        for data in val_dataloader:
            curr_batch_size = data.shape[0]

            losses = elbo_loss(said_vae, data, device)
            reconst_loss = losses["reconstruction"]
            reg_loss = losses["regularization"]
            loss = reconst_loss + beta * reg_loss

            val_total_num += curr_batch_size

            val_total_reconst_loss += reconst_loss.item() * curr_batch_size
            val_total_reg_loss += reg_loss.item() * curr_batch_size
            val_total_loss += loss.item() * curr_batch_size

    val_avg_reconst_loss = val_total_reconst_loss / val_total_num
    val_avg_reg_loss = val_total_reg_loss / val_total_num
    val_avg_loss = val_total_loss / val_total_num

    output = {
        "reconstruction": val_avg_reconst_loss,
        "regularization": val_avg_reg_loss,
        "total": val_avg_loss,
    }
    return output


def main():
    """Main function"""
    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using VOCA-ARKit dataset"
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
        default="../output",
        help="Directory of the outputs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size at training"
    )
    parser.add_argument("--epochs", type=int, default=5000, help="The number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--beta", type=float, default=0.5, help="Loss weight")
    parser.add_argument(
        "--val_period", type=int, default=50, help="Period of validating model"
    )
    parser.add_argument(
        "--save_period", type=int, default=200, help="Period of saving model"
    )
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_blendshape_coeffs_dir = os.path.join(train_dir, "blendshape_coeffs")
    val_blendshape_coeffs_dir = os.path.join(val_dir, "blendshape_coeffs")

    output_dir = args.output_dir
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    beta = args.beta
    val_period = args.val_period
    save_period = args.save_period

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", logging_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD-VAE")

    # Model
    said_vae = BCVAE()

    # Load data
    train_dataset = VOCARKitVAEDataset(train_blendshape_coeffs_dir)
    val_dataset = VOCARKitVAEDataset(val_blendshape_coeffs_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)

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
        # Train the model
        train_losses = train_epoch(
            said_vae=said_vae,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            accelerator=accelerator,
            beta=beta,
        )

        # Log
        logs = {
            "Train/Total": train_losses["total"],
            "Train/Reconst": train_losses["reconstruction"],
            "Train/Regular": train_losses["regularization"],
        }

        # Validate the model
        if epoch % val_period == 0:
            val_losses = validate_epoch(
                said_vae=said_vae,
                val_dataloader=val_dataloader,
                accelerator=accelerator,
                beta=beta,
            )
            # Append the log
            logs["Val/Total"] = val_losses["total"]
            logs["Val/Reconst"] = val_losses["reconstruction"]
            logs["Val/Regular"] = val_losses["regularization"]

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
