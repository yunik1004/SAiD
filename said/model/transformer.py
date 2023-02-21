"""Diffusion transformer
References
- https://github.com/facebookresearch/DiT/blob/main/models.py
- https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py
"""
import math
from typing import Optional
import torch
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super(TimestepEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, period, d_model)
        repeat_num = (max_seq_len // period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ConditionalDiTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
    ) -> None:
        super(ConditionalDiTBlock, self).__init__()
        self.norm_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout_sa = nn.Dropout(dropout)

        self.norm_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout_ca = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(d_model, eps=layer_norm_eps)
        dim_feedforward = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.adaLN_num_mod = 9
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, self.adaLN_num_mod * d_model, bias=True)
        )

    def forward(
        self, x: torch.FloatTensor, c: torch.FloatTensor, temb: torch.FloatTensor
    ) -> torch.FloatTensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(temb).chunk(self.adaLN_num_mod, dim=1)

        x = x + gate_msa.unsqueeze(1) * self._sa_block(
            modulate(self.norm_sa(x), shift_msa, scale_msa)
        )
        x = x + gate_mca.unsqueeze(1) * self._ca_block(
            modulate(self.norm_ca(x), shift_mca, scale_mca),
            c,
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp)
        )

        return x

    def _sa_block(
        self,
        x: torch.FloatTensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout_sa(x)

    def _ca_block(
        self,
        x: torch.FloatTensor,
        mem: torch.FloatTensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        x = self.cross_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout_ca(x)


class ConditionalDiTFinalLayer(nn.Module):
    def __init__(self, d_model: int, out_channels: int, layer_norm_eps: float = 1e-5):
        super(ConditionalDiTFinalLayer, self).__init__()
        self.norm_final = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.linear = nn.Linear(d_model, out_channels, bias=True)
        self.adaLN_num_mod = 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, self.adaLN_num_mod * d_model, bias=True),
        )

    def forward(
        self, x: torch.FloatTensor, temb: torch.FloatTensor
    ) -> torch.FloatTensor:
        shift, scale = self.adaLN_modulation(temb).chunk(self.adaLN_num_mod, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConditionalDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_in_channels: int,
        feature_dim: int,
        num_heads: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """Constructor of the ConditionalDiT

        Parameters
        ----------
        in_channels : int
            The number of channels in the input sample
        out_channels : int
            The number of channels in the output
        cond_in_channels : int
            The number of channels in the input condition
        feature_dim : int
            The dimension of the attention models
        num_heads : int
            The the number of heads in the multiheadattention models, by default 32
        num_layers : float
            The number of decoder layers, by default 1
        dropout : float
            Dropout rate, by default 0.1
        """
        super(ConditionalDiT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_in_channels = cond_in_channels
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.coeffs_encoder = nn.Linear(self.in_channels, self.feature_dim)
        self.ppe = PeriodicPositionalEncoding(self.feature_dim, dropout)
        # TODO: Change PPE -> RPE (Relative positional encoding)

        self.c_embedder = nn.Linear(self.cond_in_channels, self.feature_dim)
        self.t_embedder = TimestepEmbedder(self.feature_dim)

        self.blocks = nn.ModuleList(
            [
                ConditionalDiTBlock(self.feature_dim, self.num_heads, dropout=dropout)
                for _ in range(self.num_layers)
            ]
        )

        self.final_layer = ConditionalDiTFinalLayer(self.feature_dim, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

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
        x = self.coeffs_encoder(sample)
        x = self.ppe(x)

        temb = self.t_embedder(timestep)
        cemb = self.c_embedder(encoder_hidden_states)

        for block in self.blocks:
            x = block(x, cemb, temb)

        x = self.final_layer(x, temb)
        return x
