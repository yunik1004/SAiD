"""Define the diffusion models which are used as SAiD model
"""
from abc import ABC
from dataclasses import dataclass
import inspect
from typing import List, Optional, Union
from diffusers import DDIMScheduler
from diffusers.schedulers.scheduling_ddim import betas_for_alpha_bar
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2Processor,
)
from .unet_1d_condition import UNet1DConditionModel
from .wav2vec2 import ModifiedWav2Vec2Model


@dataclass
class SAIDInferenceOutput:
    """
    Dataclass for the inference output
    """

    result: torch.FloatTensor  # (Batch_size, sample_seq_len, x_dim), Generated blendshape coefficients
    intermediates: List[
        torch.FloatTensor
    ]  # (Batch_size, sample_seq_len, x_dim), Intermediate blendshape coefficients


@dataclass
class SAIDNoiseAdditionOutput:
    """
    Dataclass for the noise addition output
    """

    noisy_sample: torch.FloatTensor
    noise: torch.FloatTensor
    velocity: torch.FloatTensor


class SAID(ABC, nn.Module):
    """Abstract class of SAiD models"""

    denoiser: nn.Module

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        in_channels: int = 32,
        diffusion_steps: int = 1000,
        latent_scale: float = 1,
        prediction_type: str = "epsilon",
    ):
        """Constructor of SAID_UNet1D

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        in_channels : int
            Dimension of the input, by default 32
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
        prediction_type: str
            Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
        """
        super(SAID, self).__init__()

        # Audio-related
        self.audio_config = (
            audio_config if audio_config is not None else Wav2Vec2Config()
        )
        self.audio_encoder = ModifiedWav2Vec2Model(self.audio_config)
        self.audio_processor = (
            audio_processor
            if audio_processor is not None
            else Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        )
        self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate

        self.latent_scale = latent_scale

        # Noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=prediction_type,
        )

        """
        # Relieve the clipping
        self.noise_scheduler.betas = betas_for_alpha_bar(diffusion_steps, 1 - 1e-15)
        self.noise_scheduler.alphas = 1.0 - self.noise_scheduler.betas
        self.noise_scheduler.alphas_cumprod = torch.cumprod(
            self.noise_scheduler.alphas, dim=0
        )
        """

    def forward(
        self,
        noisy_samples: torch.FloatTensor,
        timesteps: torch.LongTensor,
        audio_embedding: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Return the predicted noise in the noisy samples

        Parameters
        ----------
        noisy_samples : torch.FloatTensor
            (Batch_size, coeffs_seq_len, in_channels), Sequence of noisy coefficients
        timesteps : torch.LongTensor
            (Batch_size,) or (1,), Timesteps
        audio_embedding : torch.FloatTensor
            (Batch_size, embedding_seq_len, embedding_size), Sequence of audio embeddings

        Returns
        -------
        torch.FloatTensor
            (Batch_size, coeffs_seq_len, num_coeffs), Sequence of predicted noises
        """
        timestep_size = timesteps.size()
        if len(timestep_size) == 0 or timestep_size[0] == 1:
            batch_size = noisy_samples.shape[0]
            timesteps = timesteps.repeat(batch_size)

        noise_pred = self.denoiser(noisy_samples, timesteps, audio_embedding)
        return noise_pred

    def pred_original_sample(
        self,
        noisy_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Predict the denoised sample (x_{0}) based on the noisy samples and the noise

        Parameters
        ----------
        noisy_samples : torch.FloatTensor
            (Batch_size, coeffs_seq_len, in_channels), Noisy sample
        noise : torch.FloatTensor
            (Batch_size, coeffs_seq_len, in_channels), Noise
        timesteps : torch.LongTensor
            (Batch_size,), Current timestep

        Returns
        -------
        torch.FloatTensor
            Predicted denoised sample (x_{0})
        """
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (
            noisy_samples - beta_prod_t**0.5 * noise
        ) / alpha_prod_t**0.5

        return pred_original_sample

    def process_audio(
        self, waveform: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> torch.FloatTensor:
        """Process the waveform to fit the audio encoder

        Parameters
        ----------
        waveform : Union[np.ndarray, torch.Tensor, List[np.ndarray]]
            - np.ndarray, torch.Tensor: (audio_seq_len,)
            - List[np.ndarray]: each (audio_seq_len,)

        Returns
        -------
        torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        """
        out = self.audio_processor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )["input_values"]
        return out

    def get_audio_embedding(
        self, waveform: torch.FloatTensor, num_frames: Optional[int]
    ) -> torch.FloatTensor:
        """Return the audio embedding of the waveform

        Parameters
        ----------
        waveform : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        num_frames: Optional[int]
            The length of output audio embedding sequence, by default None

        Returns
        -------
        torch.FloatTensor
            (Batch_size, embed_seq_len, embed_size), Generated audio embedding.
            If num_frames is not None, embed_seq_len = num_frames.
        """
        features = self.audio_encoder(waveform, num_frames=num_frames).last_hidden_state
        return features

    def get_random_timesteps(self, batch_size: int) -> torch.LongTensor:
        """Return the random timesteps

        Parameters
        ----------
        batch_size : int
            Size of the batch

        Returns
        -------
        torch.LongTensor
            (batch_size,), random timesteps
        """
        timesteps = torch.randint(
            0,
            self.noise_scheduler.num_train_timesteps,
            (batch_size,),
            dtype=torch.long,
        )
        return timesteps

    def add_noise(
        self, sample: torch.FloatTensor, timestep: torch.LongTensor
    ) -> SAIDNoiseAdditionOutput:
        """Add the noise into the sample

        Parameters
        ----------
        sample : torch.FloatTensor
            Sample to be noised
        timestep : torch.LongTensor
            (num_timesteps,), Timestep of the noise scheduler

        Returns
        -------
        SAIDNoiseAdditionOutput
            Noisy sample and the added noise
        """
        noise = torch.randn(sample.shape, device=sample.device)
        noisy_sample = self.noise_scheduler.add_noise(sample, noise, timestep)
        velocity = self.noise_scheduler.get_velocity(sample, noise, timestep)

        return SAIDNoiseAdditionOutput(
            noisy_sample=noisy_sample, noise=noise, velocity=velocity
        )

    def encode_samples(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        """Encode samples into latent

        Parameters
        ----------
        samples : torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Samples

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Output latent
        """
        return samples.clone()

    def decode_latent(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        """Decode latent into samples

        Parameters
        ----------
        latent : torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Latent

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, in_channels), Output samples
        """
        return latent.clone()

    def inference(
        self,
        waveform_processed: torch.FloatTensor,
        init_samples: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 100,
        strength: float = 1.0,
        guidance_scale: float = 2.5,
        eta: float = 0.0,
        fps: int = 60,
        save_intermediate: bool = False,
        show_process: bool = False,
    ) -> SAIDInferenceOutput:
        """Inference pipeline

        Parameters
        ----------
        waveform_processed : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        init_samples : Optional[torch.FloatTensor], optional
            (Batch_size, sample_seq_len, x_dim), Starting point for the process, by default None
        mask : Optional[torch.FloatTensor], optional
            (Batch_size, sample_seq_len, x_dim), Mask the region not to be changed, by default None
        num_inference_steps : int, optional
            The number of denoising steps, by default 100
        strength: float, optional
            How much to paint. Must be between 0 and 1, by default 1.0
        guidance_scale : float, optional
            Guidance scale in classifier-free guidance, by default 2.5
        eta : float, optional
            Eta in DDIM, by default 0.0
        fps : int, optional
            The number of frames per second, by default 60
        save_intermediate: bool, optional
            Return the intermediate results, by default False
        show_process: bool, optional
            Visualize the inference process, by default False

        Returns
        -------
        SAIDInferenceOutput
            Inference results and the intermediates
        """
        batch_size = waveform_processed.shape[0]
        waveform_len = waveform_processed.shape[1]
        in_channels = self.denoiser.in_channels
        device = waveform_processed.device
        do_classifier_free_guidance = guidance_scale > 1.0
        window_size = int(waveform_len / self.sampling_rate * fps)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        latents = (
            torch.randn(batch_size, window_size, in_channels, device=device)
            if init_samples is None
            else self.encode_samples(init_samples)
        )

        # Scaling the latent
        latents *= self.latent_scale * self.noise_scheduler.init_noise_sigma

        init_latents = latents.clone()
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # Add additional noise
        noise = None
        if init_samples is not None:
            timestep = self.noise_scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor(
                [timestep] * batch_size, dtype=torch.long, device=device
            )

            noise_output = self.add_noise(latents, timesteps)
            latents = noise_output.noisy_sample
            noise = noise_output.noise

        audio_embedding = self.get_audio_embedding(waveform_processed, window_size)
        if do_classifier_free_guidance:
            """
            uncond_waveform = [np.zeros((waveform_len)) for _ in range(batch_size)]
            uncond_waveform_processed = self.process_audio(uncond_waveform).to(device)
            uncond_audio_embedding = self.get_audio_embedding(
                uncond_waveform_processed, window_size
            )
            """
            uncond_audio_embedding = torch.zeros_like(audio_embedding)
            audio_embedding = torch.cat([uncond_audio_embedding, audio_embedding])

        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = eta

        intermediates = []

        t_start = num_inference_steps - init_timestep

        for idx, t in enumerate(
            tqdm(
                self.noise_scheduler.timesteps[t_start:],
                disable=not show_process,
            )
        ):
            if save_intermediate:
                interm = self.decode_latent(latents / self.latent_scale)
                intermediates.append(interm)

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            noise_pred = self.forward(latent_model_input, t, audio_embedding)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_audio + guidance_scale * (
                    noise_pred_audio - noise_pred_uncond
                )

            latents = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            # Masking
            if init_samples is not None and mask is not None:
                init_latents_noisy = init_latents

                tdx_next = t_start + idx + 1
                if tdx_next < num_inference_steps:
                    t_next = self.noise_scheduler.timesteps[tdx_next]
                    init_latents_noisy = self.noise_scheduler.add_noise(
                        init_latents, noise, t_next
                    )

                latents = init_latents_noisy * mask + latents * (1 - mask)

            # Start clipping after 90% done
            """
            if idx / init_timestep > 0.9:
                latents = (
                    self.encode_samples(
                        self.decode_latent(latents / self.latent_scale).clamp(0, 1)
                    )
                    * self.latent_scale
                )
            """

        # Re-scaling & clipping the latent
        result = self.decode_latent(latents / self.latent_scale).clamp(0, 1)

        return SAIDInferenceOutput(result=result, intermediates=intermediates)


class SAID_UNet1D(SAID):
    """SAiD model implemented using U-Net 1D model"""

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        in_channels: int = 32,
        diffusion_steps: int = 1000,
        latent_scale: float = 1,
        prediction_type: str = "epsilon",
    ):
        """Constructor of SAID_UNet1D

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        in_channels : int
            Dimension of the input, by default 32
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
        prediction_type: str
            Prediction type of the scheduler function, "epsilon", "sample", or "v_prediction", by default "epsilon"
        """
        super(SAID_UNet1D, self).__init__(
            audio_config=audio_config,
            audio_processor=audio_processor,
            in_channels=in_channels,
            diffusion_steps=diffusion_steps,
            latent_scale=latent_scale,
            prediction_type=prediction_type,
        )

        # Denoiser
        self.denoiser = UNet1DConditionModel(
            in_channels=in_channels,
            out_channels=in_channels,
            cross_attention_dim=self.audio_config.hidden_size,
        )
