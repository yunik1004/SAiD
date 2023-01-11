from typing import Union
from diffusers import UNet2DConditionModel
import torch
from torch import nn


class UNet1DConditionModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cross_attention_dim: int,
        in_hidden_channels: int = 1,
    ):
        super().__init__()
        self.conv1by1 = nn.Conv2d(1, in_hidden_channels, kernel_size=1)
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
    ):
        out = self.conv1by1(sample.unsqueeze(1)).transpose(1, 3)
        out = self.unet_2d(out, timestep, encoder_hidden_states).sample
        out = torch.mean(out, dim=3).transpose(1, 2)

        return out
