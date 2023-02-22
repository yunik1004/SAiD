"""Define the diffusion models which are used as SAiD model
"""
from abc import abstractmethod, ABC
import inspect
from typing import Dict, List, Optional, Union
from diffusers import DDPMScheduler, SchedulerMixin
import numpy as np
import torch
from torch import nn
import torchaudio
from transformers import (
    ProcessorMixin,
    PretrainedConfig,
    PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2Processor,
)
from .transformer import ConditionalDiT
from .unet_1d_condition import UNet1DConditionModel
from .vae import BCVAE
from .wav2vec2 import ModifiedWav2Vec2Model


class SAID(ABC, nn.Module):
    """Abstract class of SAiD models"""

    denoiser: nn.Module

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Optional[SchedulerMixin] = None,
        in_channels: int = 32,
        diffusion_steps: int = 1000,
        latent_scale: float = 1,
    ):
        """Constructor of SAID_UNet1D

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler : Optional[SchedulerMixin], optional
            scheduler object, by default None
        in_channels : int
            Dimension of the input, by default 32
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
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
        self.noise_scheduler = (
            noise_scheduler
            if noise_scheduler is not None
            else DDPMScheduler(
                num_train_timesteps=diffusion_steps,
                beta_start=1e-4,
                beta_end=2e-2,
                beta_schedule="squaredcos_cap_v2",
            )
        )

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
        self, samples: torch.FloatTensor, timesteps: torch.LongTensor
    ) -> Dict[str, torch.FloatTensor]:
        """Add the noise into the sample

        Parameters
        ----------
        samples : torch.FloatTensor
            Samples to be noised
        timesteps : torch.LongTensor
            (num_timesteps,), Timestep of the noise scheduler

        Returns
        -------
        Dict[str, torch.FloatTensor]
            {
                "noisy_samples": Noisy samples
                "noise": Added noise
            }
        """
        noise = torch.randn(samples.shape, device=samples.device)
        noisy_samples = self.noise_scheduler.add_noise(samples, noise, timesteps)

        output = {
            "noisy_samples": noisy_samples,
            "noise": noise,
        }
        return output

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
        num_inference_steps: int = 100,
        guidance_scale: float = 2.5,
        eta: float = 0.0,
        fps: int = 60,
        save_intermediate: bool = False,
    ) -> Dict[str, Union[torch.FloatTensor, List[torch.FloatTensor]]]:
        """Inference pipeline

        Parameters
        ----------
        waveform_processed : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        init_samples : Optional[torch.FloatTensor], optional
            (Batch_size, sample_seq_len, vae_x_dim), Starting point for the process, by default None
        num_inference_steps : int, optional
            The number of denoising steps, by default 100
        guidance_scale : float, optional
            Guidance scale in classifier-free guidance, by default 2.5
        eta : float, optional
            Eta in DDIM, by default 0.0
        fps : int, optional
            The number of frames per second, by default 60

        Returns
        -------
        Dict[str, Union[torch.FloatTensor, List[torch.FloatTensor]]]
            {
                "Result": torch.FloatTensor, (Batch_size, sample_seq_len, vae_x_dim), Generated blendshape coefficients
                "Intermediate": List[torch.FloatTensor], (Batch_size, sample_seq_len, vae_x_dim), Intermediate blendshape coefficients
            }
        """
        batch_size = waveform_processed.shape[0]
        waveform_len = waveform_processed.shape[1]
        in_channels = self.denoiser.in_channels
        device = waveform_processed.device
        do_classifier_free_guidance = guidance_scale > 1.0
        window_size = int(waveform_len / self.sampling_rate * fps)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        if init_samples is None:
            latents = torch.randn(batch_size, window_size, in_channels, device=device)
        else:
            latents = self.encode_samples(init_samples)

            # Todo: Adding additional noise would be necessary

        # Scaling the latent
        latents *= self.latent_scale * self.noise_scheduler.init_noise_sigma

        audio_embedding = self.get_audio_embedding(waveform_processed, window_size)
        if do_classifier_free_guidance:
            uncond_waveform = [np.zeros((waveform_len)) for _ in range(batch_size)]
            uncond_waveform_processed = self.process_audio(uncond_waveform).to(device)
            uncond_audio_embedding = self.get_audio_embedding(
                uncond_waveform_processed, window_size
            )

            audio_embedding = torch.cat([uncond_audio_embedding, audio_embedding])

        # Prepare extra kwargs for the scheduler step
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = eta

        intermediates = []

        for t in self.noise_scheduler.timesteps:
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
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_audio - noise_pred_uncond
                )

            latents = self.noise_scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

        # Re-scaling the latent
        result = self.decode_latent(latents / self.latent_scale)

        output = {
            "Result": result,
            "Intermediate": intermediates,
        }

        return output

    def inference_mdm(
        self,
        waveform_processed: torch.FloatTensor,
        init_samples: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 100,
        guidance_scale: float = 2.5,
        fps: int = 60,
        save_intermediate: bool = False,
    ) -> Dict[str, Union[torch.FloatTensor, List[torch.FloatTensor]]]:
        """Inference pipeline

        Parameters
        ----------
        waveform_processed : torch.FloatTensor
            (Batch_size, T_a), Processed mono waveform
        init_samples : Optional[torch.FloatTensor], optional
            (Batch_size, sample_seq_len, vae_x_dim), Starting point for the process, by default None
        num_inference_steps : int, optional
            The number of denoising steps, by default 100
        guidance_scale : float, optional
            Guidance scale in classifier-free guidance, by default 2.5
        fps : int, optional
            The number of frames per second, by default 60

        Returns
        -------
        Dict[str, Union[torch.FloatTensor, List[torch.FloatTensor]]]
            {
                "Result": torch.FloatTensor, (Batch_size, sample_seq_len, vae_x_dim), Generated blendshape coefficients
                "Intermediate": List[torch.FloatTensor], (Batch_size, sample_seq_len, vae_x_dim), Intermediate blendshape coefficients
            }
        """
        batch_size = waveform_processed.shape[0]
        waveform_len = waveform_processed.shape[1]
        in_channels = self.denoiser.in_channels
        device = waveform_processed.device
        do_classifier_free_guidance = guidance_scale > 1.0
        window_size = int(waveform_len / self.sampling_rate * fps)

        self.noise_scheduler.set_timesteps(num_inference_steps, device=device)

        if init_samples is None:
            latents = torch.randn(batch_size, window_size, in_channels, device=device)
        else:
            latents = self.encode_samples(init_samples)

            # Todo: Adding additional noise would be necessary

        # Scaling the latent
        latents *= self.latent_scale * self.noise_scheduler.init_noise_sigma

        audio_embedding = self.get_audio_embedding(waveform_processed, window_size)
        if do_classifier_free_guidance:
            uncond_waveform = [np.zeros((waveform_len)) for _ in range(batch_size)]
            uncond_waveform_processed = self.process_audio(uncond_waveform).to(device)
            uncond_audio_embedding = self.get_audio_embedding(
                uncond_waveform_processed, window_size
            )

            audio_embedding = torch.cat([uncond_audio_embedding, audio_embedding])

        intermediates = []

        for t in self.noise_scheduler.timesteps:
            if save_intermediate:
                interm = self.decode_latent(latents / self.latent_scale)
                intermediates.append(interm)

            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = self.noise_scheduler.scale_model_input(
                latent_model_input, t
            )

            latents_pred = self.forward(latent_model_input, t, audio_embedding)

            if do_classifier_free_guidance:
                latents_pred_uncond, latents_pred_audio = latents_pred.chunk(2)
                latents_pred = latents_pred_uncond + guidance_scale * (
                    latents_pred_audio - latents_pred_uncond
                )

            for temp in reversed(self.noise_scheduler.timesteps):
                if temp >= t:
                    break
                latents_pred = self.add_noise(latents_pred, temp)["noisy_samples"]

            latents = latents_pred

        # Re-scaling the latent
        result = self.decode_latent(latents / self.latent_scale)

        output = {
            "Result": result,
            "Intermediate": intermediates,
        }

        return output


class SAID_UNet1D(SAID):
    """SAiD model implemented using U-Net 1D model"""

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Optional[SchedulerMixin] = None,
        in_channels: int = 32,
        diffusion_steps: int = 1000,
        latent_scale: float = 1,
    ):
        """Constructor of SAID_UNet1D

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler : Optional[SchedulerMixin], optional
            scheduler object, by default None
        in_channels : int
            Dimension of the input, by default 32
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
        """
        super(SAID_UNet1D, self).__init__(
            audio_config=audio_config,
            audio_processor=audio_processor,
            noise_scheduler=noise_scheduler,
            in_channels=in_channels,
            diffusion_steps=diffusion_steps,
            latent_scale=latent_scale,
        )

        # Denoiser
        self.denoiser = UNet1DConditionModel(
            in_channels=in_channels,
            out_channels=in_channels,
            cross_attention_dim=self.audio_config.hidden_size,
        )


class SAID_UNet1D_LDM(SAID_UNet1D):
    """SAiD LDM implemented using U-Net 1D model"""

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Optional[SchedulerMixin] = None,
        vae_x_dim: int = 32,
        vae_h_dim: int = 16,
        vae_z_dim: int = 8,
        diffusion_steps: int = 100,
        latent_scale: float = 1,
    ):
        """Constructor of SAID_UNet1D

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler : Optional[SchedulerMixin], optional
            scheduler object, by default None
        vae_x_dim : int
            Dimension of the input, by default 32
        vae_h_dim : int
            Dimension of the hidden layer, by default 16
        vae_z_dim : int
            Dimension of the latent, by default 8
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
        """
        super(SAID_UNet1D_LDM, self).__init__(
            audio_config=audio_config,
            audio_processor=audio_processor,
            noise_scheduler=noise_scheduler,
            in_channels=z_dim,
            diffusion_steps=diffusion_steps,
            latent_scale=latent_scale,
        )

        # VAE
        self.vae = BCVAE(x_dim=vae_x_dim, h_dim=vae_h_dim, z_dim=vae_z_dim)

    def encode_samples(self, samples: torch.FloatTensor) -> torch.FloatTensor:
        """Encode samples into latent

        Parameters
        ----------
        samples : torch.FloatTensor
            (Batch_size, sample_seq_len, vae_x_dim), Samples

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, vae_z_dim), Output latent
        """
        latent_stats = self.vae.encode(samples)
        latents = self.vae.reparametrize(
            latent_stats["mean"], latent_stats["log_var"], True
        )
        return latents

    def decode_latent(self, latent: torch.FloatTensor) -> torch.FloatTensor:
        """Decode latent into samples

        Parameters
        ----------
        latent : torch.FloatTensor
            (Batch_size, sample_seq_len, vae_z_dim), Latent

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, vae_x_dim), Output samples
        """
        return self.vae.decode(latent)


class SAID_CDiT(SAID):
    """SAiD model implemented using Conditional DiT model"""

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Optional[SchedulerMixin] = None,
        in_channels: int = 32,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        diffusion_steps: int = 1000,
        latent_scale: float = 1,
    ):
        """Constructor of SAID_CDiT

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler : Optional[SchedulerMixin], optional
            scheduler object, by default None
        in_channels : int
            Dimension of the input, by default 32
        feature_dim : int
            Dimension of the model feature, by default 256
        num_heads : int
            The number of heads in transformer, by default 4
        num_layers : int
            The number of transformer layers, by default 8
        diffusion_steps : int
            The number of diffusion steps, by default 1000
        latent_scale : float
            Scaling the latent, by default 1
        """
        super(SAID_CDiT, self).__init__(
            audio_config=audio_config,
            audio_processor=audio_processor,
            noise_scheduler=noise_scheduler,
            in_channels=in_channels,
            diffusion_steps=diffusion_steps,
            latent_scale=latent_scale,
        )
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Denoiser
        self.denoiser = ConditionalDiT(
            in_channels=in_channels,
            out_channels=in_channels,
            cond_in_channels=self.audio_config.hidden_size,
            feature_dim=self.feature_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        )
