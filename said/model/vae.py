"""Define the autoencoder for the blendshape coefficients
"""
from typing import Dict
import torch
from torch import nn


class BCEncoder(nn.Module):
    """Encoder for the blendshape coefficients"""

    def __init__(self, x_dim: int, h_dim: int, z_dim: int) -> None:
        """Constructor of the BCEncoder

        Parameters
        ----------
        x_dim : int
            Dimension of the input
        h_dim : int
            Dimension of the hidden layer
        z_dim : int
            Dimension of the latent
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.Mish(),
            nn.Linear(h_dim, h_dim),
            nn.Mish(),
            nn.Linear(h_dim, 2 * z_dim),
        )

    def forward(self, coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward function

        Parameters
        ----------
        coeffs : torch.Tensor
            (Batch_size, sample_seq_len, x_dim), Blendshape coefficients

        Returns
        -------
        Dict[str, torch.Tensor]
            {
                "mean": (Batch_size, sample_seq_len, z_dim), mean
                "log_var": (Batch_size, sample_seq_len, z_dim), log of the variance
            }
        """
        batch_size, sample_seq_len, x_dim = coeffs.shape
        latent = self.layers(coeffs.reshape(-1, x_dim))
        latent = latent.reshape(batch_size, sample_seq_len, -1)

        mean, log_var = torch.chunk(latent, 2, dim=-1)

        output = {
            "mean": mean,
            "log_var": log_var,
        }
        return output

    @staticmethod
    def reparametrize(
        mean: torch.FloatTensor,
        log_var: torch.FloatTensor,
        align_noise: bool = False,
    ) -> torch.FloatTensor:
        """Reparametrize the latent using mean, log_var

        Parameters
        ----------
        mean : torch.FloatTensor
            (Batch_size, sample_seq_len, z_dim), mean
        log_var : torch.FloatTensor
            (Batch_size, sample_seq_len, z_dim), log of the variance
        align_noise : bool, optional
            Whether the noises are the same in each batch, by default False

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, z_dim), Latent vectors (sampled)
        """
        batch_size, sample_seq_len, _ = mean.shape
        device = mean.device
        eps = (
            torch.randn(batch_size, 1, 1)
            if align_noise
            else torch.randn(batch_size, sample_seq_len, 1)
        ).to(device)
        latent = mean + torch.exp(0.5 * log_var) * eps
        return latent


class BCDecoder(nn.Module):
    """Decoder for the blendshape coefficients"""

    def __init__(self, z_dim: int, h_dim: int, x_dim: int) -> None:
        """Constructor of BCDecoder

        Parameters
        ----------
        z_dim : int
            Dimension of the latent
        h_dim : int
            Dimension of the hidden layer
        x_dim : int
            Dimension of the input
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.Mish(),
            nn.Linear(h_dim, h_dim),
            nn.Mish(),
            nn.Linear(h_dim, x_dim),
            nn.Tanh(),
            nn.ReLU(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward function

        Parameters
        ----------
        latent : torch.Tensor
            (Batch_size, sample_seq_len, z_dim), Latent vectors (sampled)

        Returns
        -------
        torch.Tensor
            (Batch_size, sample_seq_len, x_dim), Reconstructed blendshape coefficients
        """
        batch_size, sample_seq_len, z_dim = latent.shape
        coeffs_reconst = self.layers(latent.reshape(-1, z_dim))
        coeffs_reconst = coeffs_reconst.reshape(batch_size, sample_seq_len, -1)
        return coeffs_reconst


class BCVAE(nn.Module):
    """Autoencoder for the blendshape coefficients"""

    def __init__(
        self,
        x_dim: int = 32,
        h_dim: int = 16,
        z_dim: int = 8,
    ) -> None:
        """Constructor of the BCAutoEncoder

        Parameters
        ----------
        x_dim : int, optional
            Dimension of the input, by default 32
        h_dim : int, optional
            Dimension of the hidden layer, by default 16
        z_dim : int, optional
            Dimension of the latent, by default 8
        """
        super().__init__()
        self.encoder = BCEncoder(x_dim, h_dim, z_dim)
        self.decoder = BCDecoder(z_dim, h_dim, x_dim)

    def forward(
        self, coeffs: torch.Tensor, use_noise: bool = True, align_noise: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward function

        Parameters
        ----------
        coeffs : torch.Tensor
            (Batch_size, sample_seq_len, x_dim), Blendshape coefficients
        use_noise : bool, optional
            Whether using noises when reconstructing the coefficients, by default True
        align_noise : bool, optional
            Whether the noises are the same in each batch, by default False

        Returns
        -------
        Dict[str, torch.Tensor]
            {
                "mean": (Batch_size, sample_seq_len, z_dim), mean (latent)
                "log_var": (Batch_size, sample_seq_len, z_dim), log of the variance (latent)
                "latent": (Batch_size, sample_seq_len, z_dim), latent (sampled)
                "reconstruction": (Batch_size, sample_seq_len, x_dim), Reconstructed blendshape coefficients
            }
        """
        latent_dict = self.encoder(coeffs)
        mean = latent_dict["mean"]
        log_var = latent_dict["log_var"]
        latent = (
            self.encoder.reparametrize(mean, log_var, align_noise)
            if use_noise
            else mean
        )
        coeffs_reconst = self.decoder(latent)
        output = {
            "mean": mean,
            "log_var": log_var,
            "latent": latent,
            "reconstruction": coeffs_reconst,
        }
        return output

    def encode(self, coeffs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode the blendshape coefficients

        Parameters
        ----------
        coeffs : torch.Tensor
            (Batch_size, sample_seq_len, x_dim), Blendshape coefficients

        Returns
        -------
        Dict[str, torch.Tensor]
            {
                "mean": (Batch_size, sample_seq_len, z_dim), mean
                "log_var": (Batch_size, sample_seq_len, z_dim), log of the variance
            }
        """
        return self.encoder(coeffs)

    def reparametrize(
        self,
        mean: torch.FloatTensor,
        log_var: torch.FloatTensor,
        align_noise: bool = False,
    ) -> torch.FloatTensor:
        """Reparametrize the latent using mean, log_var

        Parameters
        ----------
        mean : torch.FloatTensor
            (Batch_size, sample_seq_len, z_dim), mean
        log_var : torch.FloatTensor
            (Batch_size, sample_seq_len, z_dim), log of the variance
        align_noise : bool, optional
            Whether the noises are the same in each batch, by default False

        Returns
        -------
        torch.FloatTensor
            (Batch_size, sample_seq_len, z_dim), Latent vectors (sampled)
        """
        return self.encoder.reparametrize(mean, log_var, align_noise)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector into blendshape coefficients

        Parameters
        ----------
        latent : torch.Tensor
            (Batch_size, sample_seq_len, z_dim), Latent vectors (sampled)

        Returns
        -------
        torch.Tensor
            (Batch_size, sample_seq_len, x_dim), Blendshape coefficients
        """
        return self.decoder(latent)
