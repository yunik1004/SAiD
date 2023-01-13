"""Define the diffusion models which are used as SAiD model
"""
from abc import abstractmethod, ABC
from typing import List, Optional, Tuple, Union
from diffusers import DDIMScheduler, SchedulerMixin
import numpy as np
import torch
from torch import nn
import torchaudio
from transformers import (
    ProcessorMixin,
    PretrainedConfig,
    PreTrainedModel,
    Wav2Vec2Config,
    Wav2Vec2Model,
    Wav2Vec2Processor,
)
from .unet_1d_condition import UNet1DConditionModel


class SAID(ABC, nn.Module):
    """Abstract class of SAiD models"""

    audio_encoder: nn.Module
    audio_processor: ProcessorMixin
    sampling_rate: int
    denoiser: nn.Module
    noise_scheduler: SchedulerMixin

    def process_audio(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        """Process the waveform to fit the audio encoder

        Parameters
        ----------
        waveform : torch.FloatTensor
            (T_a), Mono waveform

        Returns
        -------
        torch.FloatTensor
            (T_a), Processed mono waveform
        """
        out = self.audio_processor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        )["input_values"].squeeze(0)
        return out

    @abstractmethod
    def get_audio_embedding(self, waveform: torch.FloatTensor) -> torch.FloatTensor:
        """Return the audio embedding of the waveform

        Parameters
        ----------
        waveform : Union[np.ndarray, torch.Tensor, List[np.ndarray]]
            - np.ndarray, torch.Tensor: (audio_seq_len,)
            - List[np.ndarray]: each (audio_seq_len,)

        Returns
        -------
        torch.FloatTensor
            (Batch_size, embed_seq_len, embed_size), Generated audio embedding
        """
        pass

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
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Add the noise into the sample

        Parameters
        ----------
        samples : torch.FloatTensor
            Samples to be noised
        timesteps : torch.LongTensor
            (num_timesteps,), Timestep of the noise scheduler

        Returns
        -------
        Tuple[torch.FloatTensor, torch.FloatTensor]
            Noised samples, and Noise
        """
        noise = torch.randn(samples.shape, device=samples.device)
        noisy_samples = self.noise_scheduler.add_noise(samples, noise, timesteps)
        return noisy_samples, noise


class SAID_Wav2Vec2(SAID):
    """SAiD model implemented using Wav2Vec2 audio model"""

    def __init__(
        self,
        audio_config: Optional[Wav2Vec2Config] = None,
        audio_processor: Optional[Wav2Vec2Processor] = None,
        noise_scheduler: Optional[SchedulerMixin] = None,
        in_channels: int = 32,
    ):
        """Constructor of SAID_Wav2Vec2

        Parameters
        ----------
        audio_config : Optional[Wav2Vec2Config], optional
            Wav2Vec2Config object, by default None
        audio_processor : Optional[Wav2Vec2Processor], optional
            Wav2Vec2Processor object, by default None
        noise_scheduler : Optional[SchedulerMixin], optional
            scheduler object, by default None
        in_channels: int
            Size of the input channel, by default 32
        """
        super(SAID_Wav2Vec2, self).__init__()

        # Audio-related
        self.audio_config = (
            audio_config if audio_config is not None else Wav2Vec2Config()
        )
        self.audio_encoder = Wav2Vec2Model(self.audio_config)
        self.audio_processor = (
            audio_processor
            if audio_processor is not None
            else Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        )
        self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate

        # Denoiser-related
        self.denoiser = UNet1DConditionModel(
            in_channels=in_channels,
            out_channels=in_channels,
            cross_attention_dim=self.audio_config.conv_dim[-1],
        )
        self.noise_scheduler = (
            noise_scheduler if noise_scheduler is not None else DDIMScheduler()
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
            (Batch_size, coeffs_seq_len, num_coeffs), Sequence of noisy coefficients
        timesteps : torch.LongTensor
            (Batch_size,), Timesteps
        audio_embedding : torch.FloatTensor
            (Batch_size, embedding_seq_len, embedding_size), Sequence of audio embeddings

        Returns
        -------
        torch.FloatTensor
            (Batch_size, coeffs_seq_len, num_coeffs), Sequence of predicted noises
        """
        noise_pred = self.denoiser(noisy_samples, timesteps, audio_embedding)
        return noise_pred

    def get_audio_embedding(
        self, waveform: Union[np.ndarray, torch.Tensor, List[np.ndarray]]
    ) -> torch.FloatTensor:
        features = self.audio_encoder(waveform).extract_features
        return features
