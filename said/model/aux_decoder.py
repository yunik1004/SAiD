"""Define the auxiliary decoder
"""
import torch
from torch import nn
from ..ldm.attention import SpatialTransformer
from .transformer import PositionalEncoding


class AuxDecoder(nn.Module):
    """Auxiliary decoder module for blendshape coefficients generator"""

    def __init__(
        self,
        in_channels: int = 768,
        out_channels: int = 32,
        feature_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 4,
        num_groups: int = 32,
        dropout: float = 0.1,
    ) -> None:
        """Constructor of the AuxDecoder

        Parameters
        ----------
        in_channels : int, optional
            Input dimension, by default 768
        out_channels : int, optional
            Output dimension, by default 32
        feature_dim: int, optional
            Feature dimension, by default 256
        num_layers : int, optional
            The number of transformer layers, by default 1
        num_heads : int, optional
            The number of heads, by default 4
        num_groups: int, optional
            The number of groups in group norm, by default 32
        dropout : float, optional
            Dropout probability, by default 0.1
        """
        super(AuxDecoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_groups = num_groups
        dim_feedforward = self.feature_dim * 4

        self.enc_layer = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.feature_dim,
            kernel_size=3,
            padding=1,
        )

        self.pe = PositionalEncoding(self.feature_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

        self.norm = nn.GroupNorm(
            num_groups=self.num_groups, num_channels=self.feature_dim
        )
        self.silu = nn.SiLU()

        self.dec_layer = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """Return the expected blendshape coefficients

        Parameters
        ----------
        encoder_hidden_states : torch.FloatTensor
            (Batch_size, hidden_states_seq_len, in_channels), Audio features

        Returns
        -------
        torch.FloatTensor
            (Batch_size, hidden_states_seq_len, out_channels), Predicted blendshape coefficients
        """
        x = self.enc_layer(encoder_hidden_states.transpose(1, 2))
        x = self.pe(x.permute(2, 0, 1)).transpose(0, 1)
        x = self.transformer(x).transpose(1, 2)
        x = self.dec_layer(self.silu(self.norm(x))).transpose(1, 2)

        return x
