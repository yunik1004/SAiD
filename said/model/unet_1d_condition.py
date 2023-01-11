"""Define the conditional 1D UNet model
"""
from typing import Union
from diffusers import UNet2DConditionModel
import torch
from torch import nn


class UNet1DConditionModel(nn.Module):
    """Conditional 1D UNet model"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cross_attention_dim: int,
    ):
        """Constructor of the UNet1DConditionModel

        Parameters
        ----------
        in_channels : int
            The number of channels in the input sample
        out_channels : int
            The number of channels in the output
        cross_attention_dim : int
            The dimension of the cross attention features
        """
        super().__init__()
        self.unet_2d = UNet2DConditionModel(
            in_channels=in_channels,
            out_channels=out_channels,
            cross_attention_dim=cross_attention_dim,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        """Denoise the input sample

        Parameters
        ----------
        sample : torch.FloatTensor
            (Batch_size, sample_seq_len, channel), Noisy inputs tensor
        timestep : Union[torch.Tensor, float, int]
            (Batch_size,), Timesteps
        encoder_hidden_states : torch.Tensor
            (Batch_size, hidden_states_seq_len, cross_attention_dim), Encoder hidden states

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, channel), Predicted noise
        """
        out = sample.unsqueeze(1).transpose(1, 3)
        out = self.unet_2d(out, timestep, encoder_hidden_states).sample
        out = out.squeeze(3).transpose(1, 2)

        return out
