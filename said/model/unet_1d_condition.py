"""Define the conditional 1D UNet model
"""
import torch
from torch import nn
from .ldm.openaimodel import UNetModel


class UNet1DConditionModel(nn.Module):
    """Conditional 1D UNet model"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cross_attention_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Constructor of the UNet1DConditionModel

        Parameters
        ----------
        in_channels : int
            The number of channels in the input sample
        out_channels : int
            The number of channels in the output
        cross_attention_dim : int
            The dimension of the cross attention features
        dropout : float
            Dropout ratte, by default 0.1
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim

        self.model = UNetModel(
            dims=1,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            model_channels=192,
            num_res_blocks=1,
            attention_resolutions=(1,),
            dropout=dropout,
            channel_mult=(1,),
            num_head_channels=32,
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=self.cross_attention_dim,
        )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.FloatTensor:
        """Denoise the input sample

        Parameters
        ----------
        sample : torch.FloatTensor
            (Batch_size, sample_seq_len, channel), Noisy inputs tensor
        timestep : torch.Tensor
            (Batch_size,), (1,), or (), Timesteps
        encoder_hidden_states : torch.Tensor
            (Batch_size, hidden_states_seq_len, cross_attention_dim), Encoder hidden states

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, channel), Predicted noise
        """
        out = sample.transpose(1, 2)
        out = self.model(out, timestep, encoder_hidden_states)
        out = out.transpose(1, 2)

        return out
