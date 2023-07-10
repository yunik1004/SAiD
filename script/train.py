"""Train the SAID_UNet1D model
"""
import argparse
from dataclasses import dataclass
import os
import pathlib
from typing import Optional
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from said.model.diffusion import SAID, SAID_UNet1D
from said.model.wav2vec2 import ModifiedWav2Vec2Model
from dataset.dataset_voca import DataBatch, VOCARKitTrainDataset, VOCARKitValDataset


@dataclass
class LossStepOutput:
    """
    Dataclass for the losses at each step
    """

    predict: torch.FloatTensor  # MAE loss for the predicted output
    velocity: torch.FloatTensor  # MAE loss for the velocity
    vertex: Optional[torch.FloatTensor]  # MAE loss for the reconstructed vertex


@dataclass
class LossEpochOutput:
    """
    Dataclass for the averaged losses at each epoch
    """

    total: float = 0  # Averaged total loss
    predict: float = 0  # Averaged prediction loss
    velocity: float = 0  # Averaged velocity loss
    vertex: float = 0  # Averaged vertex loss


def random_noise_loss(
    said_model: SAID,
    data: DataBatch,
    device: torch.device,
    prediction_type: str = "epsilon",
) -> LossStepOutput:
    """Compute the loss with randomized noises

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    data : DataBatch
        Output of the VOCARKitDataset.collate_fn
    device : torch.device
        GPU device
    prediction_type: str
        Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"

    Returns
    -------
    LossStepOutput
        Computed losses
    """
    waveform = data.waveform
    blendshape_coeffs = data.blendshape_coeffs.to(device)
    blendshape_delta = data.blendshape_delta.to(device)
    cond = data.cond.to(device)

    coeff_latents = said_model.encode_samples(
        blendshape_coeffs * said_model.latent_scale
    )

    curr_batch_size = len(waveform)
    window_size = blendshape_coeffs.shape[1]

    waveform_processed = said_model.process_audio(waveform).to(device)
    random_timesteps = said_model.get_random_timesteps(curr_batch_size).to(device)

    audio_embedding = said_model.get_audio_embedding(waveform_processed, window_size)
    cond_mask = audio_embedding * cond.view(-1, 1, 1)
    uncond_embedding = said_model.null_cond_emb.repeat(curr_batch_size, window_size, 1)
    audio_embedding_cond = (
        audio_embedding * cond_mask + uncond_embedding * torch.logical_not(cond_mask)
    )

    noise_dict = said_model.add_noise(coeff_latents, random_timesteps)
    noisy_latents = noise_dict.noisy_sample
    noise = noise_dict.noise
    velocity = noise_dict.velocity

    pred = said_model(noisy_latents, random_timesteps, audio_embedding_cond)

    # Set answer corresponding to prediction_type
    answer = None
    if prediction_type == "epsilon":
        answer = noise
    elif prediction_type == "sample":
        answer = coeff_latents
    elif prediction_type == "v_prediction":
        answer = velocity

    criterion_pred = nn.L1Loss()
    criterion_velocity = nn.L1Loss()
    criterion_vertex = nn.L1Loss()

    loss_pred = criterion_pred(answer, pred)

    answer_diff = answer[:, 1:, :] - answer[:, :-1, :]
    pred_diff = pred[:, 1:, :] - pred[:, :-1, :]

    loss_vel = criterion_velocity(answer_diff, pred_diff)

    loss_vertex = None
    if blendshape_delta is not None:
        b, k, v, i = blendshape_delta.shape
        _, t, _ = answer.shape

        blendshape_delta_norm = torch.norm(blendshape_delta, p=1, dim=[1, 2, 3]) / (
            k * v * i
        )
        blendshape_delta_normalized = torch.div(
            blendshape_delta,
            blendshape_delta_norm.view(-1, 1, 1, 1),
        )

        be_answer = torch.bmm(answer, blendshape_delta_normalized.view(b, k, v * i))
        be_pred = torch.bmm(pred, blendshape_delta_normalized.view(b, k, v * i))

        # be_answer = torch.einsum("bkvi,btk->btvi", blendshape_delta_normalized, answer)
        # be_pred = torch.einsum("bkvi,btk->btvi", blendshape_delta_normalized, pred)

        loss_vertex = criterion_vertex(be_answer, be_pred)

    return LossStepOutput(
        predict=loss_pred,
        velocity=loss_vel,
        vertex=loss_vertex,
    )


def train_epoch(
    said_model: SAID,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    weight_vel: float,
    weight_vertex: float,
    prediction_type: str = "epsilon",
    ema_model: Optional[EMAModel] = None,
) -> LossEpochOutput:
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
    weight_vel: float
        Weight for the velocity loss
    weight_vertex: float
        Weight for the vertex loss
    prediction_type: str
        Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
    ema_model: Optional[EMAModel]
        EMA model of said_model, by default None

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = accelerator.device

    said_model.train()

    train_total_losses = {
        "loss": 0,
        "loss_predict": 0,
        "loss_velocity": 0,
        "loss_vertex": 0,
    }
    train_total_num = 0
    for data in train_dataloader:
        curr_batch_size = len(data.waveform)
        losses = random_noise_loss(said_model, data, device, prediction_type)

        loss = losses.predict + weight_vel * losses.velocity
        if losses.vertex is not None:
            loss += weight_vertex * losses.vertex

        accelerator.backward(loss)
        optimizer.step()
        if ema_model:
            ema_model.step(said_model.parameters())
        optimizer.zero_grad()

        train_total_losses["loss"] += loss.item() * curr_batch_size
        train_total_losses["loss_predict"] += losses.predict.item() * curr_batch_size
        train_total_losses["loss_velocity"] += losses.velocity.item() * curr_batch_size
        if losses.vertex is not None:
            train_total_losses["loss_vertex"] += losses.vertex.item() * curr_batch_size

        train_total_num += curr_batch_size

    train_avg_losses = LossEpochOutput(
        total=train_total_losses["loss"] / train_total_num,
        predict=train_total_losses["loss_predict"] / train_total_num,
        velocity=train_total_losses["loss_velocity"] / train_total_num,
        vertex=train_total_losses["loss_vertex"] / train_total_num,
    )

    return train_avg_losses


def validate_epoch(
    said_model: SAID,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    weight_vel: float,
    weight_vertex: float,
    prediction_type: str = "epsilon",
    num_repeat: int = 1,
) -> LossEpochOutput:
    """Validate the SAiD model one epoch.

    Parameters
    ----------
    said_model : SAID
        SAiD model object
    val_dataloader : DataLoader
        Dataloader of the VOCARKitValDataset
    accelerator : Accelerator
        Accelerator object
    weight_vel: float
        Weight for the velocity loss
    weight_vertex: float
        Weight for the vertex loss
    prediction_type: str
        Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
    num_repeat : int, optional
        Number of the repetition, by default 1

    Returns
    -------
    LossEpochOutput
        Average losses
    """
    device = accelerator.device

    said_model.eval()

    val_total_losses = {
        "loss": 0,
        "loss_predict": 0,
        "loss_velocity": 0,
        "loss_vertex": 0,
    }
    val_total_num = 0
    with torch.no_grad():
        for _ in range(num_repeat):
            for data in val_dataloader:
                curr_batch_size = len(data.waveform)
                losses = random_noise_loss(said_model, data, device, prediction_type)

                loss = losses.predict + weight_vel * losses.velocity
                if losses.vertex is not None:
                    loss += weight_vertex * losses.vertex

                val_total_losses["loss"] += loss.item() * curr_batch_size
                val_total_losses["loss_predict"] += (
                    losses.predict.item() * curr_batch_size
                )
                val_total_losses["loss_velocity"] += (
                    losses.velocity.item() * curr_batch_size
                )
                if losses.vertex is not None:
                    val_total_losses["loss_vertex"] += (
                        losses.vertex.item() * curr_batch_size
                    )

                val_total_num += curr_batch_size

    val_avg_losses = LossEpochOutput(
        total=val_total_losses["loss"] / val_total_num,
        predict=val_total_losses["loss_predict"] / val_total_num,
        velocity=val_total_losses["loss_velocity"] / val_total_num,
        vertex=val_total_losses["loss_vertex"] / val_total_num,
    )

    return val_avg_losses


def main() -> None:
    """Main function"""
    default_data_dir = pathlib.Path(__file__).parent.parent / "data"

    # Arguments
    parser = argparse.ArgumentParser(
        description="Train the SAiD model using VOCA-ARKit dataset"
    )
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
        "--blendshape_residuals_path",
        type=str,
        default=(default_data_dir / "blendshape_residuals.pickle").resolve(),
        help="Path of the blendshape residuals",
    )
    parser.add_argument(
        "--landmarks_path",
        type=str,
        default=(default_data_dir / "FLAME_head_landmarks.txt").resolve(),
        help="Path of the landmarks data",
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
        "--window_size",
        type=int,
        default=120,
        help="Window size of the blendshape coefficients sequence at training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size at training"
    )
    parser.add_argument(
        "--epochs", type=int, default=30000, help="The number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--uncond_prob",
        type=float,
        default=0.1,
        help="Unconditional probability of waveform (for classifier-free guidance)",
    )
    parser.add_argument(
        "--weight_vel",
        type=float,
        default=1.0,
        help="Weight for the velocity loss",
    )
    parser.add_argument(
        "--weight_vertex",
        type=float,
        default=0.02,
        help="Weight for the vertex loss",
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
        "--val_period", type=int, default=200, help="Period of validating model"
    )
    parser.add_argument(
        "--val_repeat", type=int, default=50, help="Number of repetition of val dataset"
    )
    parser.add_argument(
        "--save_period", type=int, default=200, help="Period of saving model"
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir
    coeffs_dir = args.coeffs_dir
    blendshape_deltas_path = args.blendshape_residuals_path
    if blendshape_deltas_path == "":
        blendshape_deltas_path = None
    landmarks_path = args.landmarks_path
    if landmarks_path == "":
        landmarks_path = None

    output_dir = args.output_dir
    prediction_type = args.prediction_type
    window_size = args.window_size
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    uncond_prob = args.uncond_prob
    weight_vel = args.weight_vel
    weight_vertex = args.weight_vertex
    ema = args.ema
    ema_decay = args.ema_decay
    val_period = args.val_period
    val_repeat = args.val_repeat
    save_period = args.save_period

    # Initialize accelerator
    accelerator = Accelerator(log_with="tensorboard", project_dir=output_dir)
    if accelerator.is_main_process:
        accelerator.init_trackers("SAiD")

    said_model = SAID_UNet1D()
    said_model.audio_encoder = ModifiedWav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base-960h"
    )

    # Load data
    train_dataset = VOCARKitTrainDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_dir,
        blendshape_deltas_path=blendshape_deltas_path,
        landmarks_path=landmarks_path,
        sampling_rate=said_model.sampling_rate,
        window_size=window_size,
        uncond_prob=uncond_prob,
        preload=True,
    )
    val_dataset = VOCARKitValDataset(
        audio_dir=audio_dir,
        blendshape_coeffs_dir=coeffs_dir,
        blendshape_deltas_path=blendshape_deltas_path,
        landmarks_path=landmarks_path,
        sampling_rate=said_model.sampling_rate,
        uncond_prob=uncond_prob,
        preload=True,
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

    # Initialize the optimzier - freeze audio encoder
    for p in said_model.audio_encoder.parameters():
        p.requires_grad = False

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
    ema_model = EMAModel(said_model.parameters(), decay=ema_decay) if ema else None

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
            weight_vel=weight_vel,
            weight_vertex=weight_vertex,
            prediction_type=prediction_type,
            ema_model=ema_model,
        )

        # Log
        logs = {
            "Train/Total Loss": train_avg_losses.total,
            "Train/Predict Loss": train_avg_losses.predict,
            "Train/Velocity Loss": train_avg_losses.velocity,
            "Train/Vertex Loss": train_avg_losses.vertex,
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
                weight_vel=weight_vel,
                weight_vertex=weight_vertex,
                prediction_type=prediction_type,
                num_repeat=val_repeat,
            )
            # Append the log
            logs["Validation/Total Loss"] = val_avg_losses.total
            logs["Validation/Predict Loss"] = val_avg_losses.predict
            logs["Validation/Velocity Loss"] = val_avg_losses.velocity
            logs["Validation/Vertex Loss"] = val_avg_losses.vertex

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
