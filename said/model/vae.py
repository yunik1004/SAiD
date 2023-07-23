"""Define the autoencoder for the blendshape coefficients
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class BCLatent:
    """Latent variables of the BCVAE"""

    mean: torch.FloatTensor
    log_var: torch.FloatTensor


@dataclass
class BCVAEOutput:
    """Output of the BCVAE"""

    mean: torch.FloatTensor
    log_var: torch.FloatTensor
    latent: torch.FloatTensor
    coeffs_reconst: torch.FloatTensor


class BCEncoder(nn.Module):
    """Encoder for the blendshape coefficients"""

    def __init__(self, in_channels: int = 32, z_dim: int = 64) -> None:
        """Constructor of the BCEncoder

        Parameters
        ----------
        in_channels : int
            The number of input channels, by default 32
        z_dim : int
            Dimension of the latent, by default 64
        """
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 64, kernel_size=3, stride=1),
            nn.Flatten(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3520, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, z_dim),
        )

        self.fc_mu = nn.Linear(z_dim, z_dim)
        self.fc_logvar = nn.Linear(z_dim, z_dim)

    def forward(self, coeffs: torch.Tensor) -> BCLatent:
        """Forward function

        Parameters
        ----------
        coeffs : torch.Tensor
            (Batch_size, sample_seq_len=120, in_channels), Blendshape coefficients

        Returns
        -------
        BCLatent
            Latent variables, including mean and log_var
        """
        out = self.conv_layers(coeffs.transpose(1, 2))
        latent = self.fc_layers(out)

        mean = self.fc_mu(latent)
        log_var = self.fc_logvar(latent)

        return BCLatent(mean=mean, log_var=log_var)

    @staticmethod
    def reparametrize(
        mean: torch.FloatTensor,
        log_var: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Reparametrize the latent using mean, log_var

        Parameters
        ----------
        mean : torch.FloatTensor
            (Batch_size, z_dim), mean
        log_var : torch.FloatTensor
            (Batch_size, z_dim), log of the variance

        Returns
        -------
        torch.FloatTensor
            (Batch_size, z_dim), Latent vectors (sampled)
        """
        batch_size, z_dim = mean.shape
        device = mean.device
        eps = torch.randn(batch_size, z_dim).to(device)
        latent = mean + torch.exp(0.5 * log_var) * eps
        return latent


class BCDecoder(nn.Module):
    """Decoder for the blendshape coefficients"""

    def __init__(
        self, out_channels: int = 32, seq_len: int = 120, z_dim: int = 64
    ) -> None:
        """Constructor of BCDecoder

        Parameters
        ----------
        out_channels : int
            The number of channels of the output, by default 32
        seq_len : int
            Lenght of the output sequence, by default 120
        z_dim : int
            Dimension of the latent, by default 64
        """
        super().__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(z_dim, 2 * seq_len),
            nn.BatchNorm1d(2 * seq_len),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * seq_len, 4 * seq_len),
            nn.Unflatten(1, (4, seq_len)),
        )

        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(4, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose1d(32, 32, 3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(32, 32, 3),
            nn.Conv1d(32, out_channels, 3),
            nn.ReLU(),
            nn.Tanh(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Forward function

        Parameters
        ----------
        latent : torch.Tensor
            (Batch_size, z_dim), Latent vectors

        Returns
        -------
        torch.Tensor
            (Batch_size, sample_seq_len, out_channels), Reconstructed blendshape coefficients
        """
        out = self.fc_layers(latent)
        coeffs_reconst = self.conv_layers(out).transpose(1, 2)
        return coeffs_reconst


class BCVAE(nn.Module):
    """Autoencoder for the blendshape coefficients"""

    def __init__(
        self,
        channels: int = 32,
        seq_len: int = 120,
        z_dim: int = 64,
    ) -> None:
        """Constructor of the BCAutoEncoder

        Parameters
        ----------
        channels : int
            The number of channels of the input and output, by default 32
        seq_len : int
            Lenght of the output sequence, by default 120
        z_dim : int
            Dimension of the latent, by default 64
        """
        super().__init__()
        self.seq_len = seq_len
        self.encoder = BCEncoder(channels, z_dim)
        self.decoder = BCDecoder(channels, seq_len, z_dim)

    def forward(self, coeffs: torch.Tensor, use_noise: bool = True) -> BCVAEOutput:
        """Forward function

        Parameters
        ----------
        coeffs : torch.Tensor
            (Batch_size, sample_seq_len, x_dim), Blendshape coefficients
        use_noise : bool, optional
            Whether using noises when reconstructing the coefficients, by default True

        Returns
        -------
        BCVAEOutput
            Output of the VAE including the latent variables
        """
        latent_stats = self.encode(coeffs)
        mean = latent_stats.mean
        log_var = latent_stats.log_var
        latent = self.reparametrize(mean, log_var) if use_noise else mean
        coeffs_reconst = self.decode(latent)

        return BCVAEOutput(
            mean=mean, log_var=log_var, latent=latent, coeffs_reconst=coeffs_reconst
        )

    def encode(self, coeffs: torch.Tensor) -> BCLatent:
        """Encode the blendshape coefficients

        Parameters
        ----------
        coeffs : torch.Tensor
            (Batch_size, sample_seq_len, x_dim), Blendshape coefficients

        Returns
        -------
        BCLatent
            Latent variables of the VAE
        """
        return self.encoder(coeffs)

    def reparametrize(
        self,
        mean: torch.FloatTensor,
        log_var: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Reparametrize the latent using mean, log_var

        Parameters
        ----------
        mean : torch.FloatTensor
            (Batch_size, z_dim), mean
        log_var : torch.FloatTensor
            (Batch_size, z_dim), log of the variance

        Returns
        -------
        torch.FloatTensor
            (Batch_size, z_dim), Latent vectors (sampled)
        """
        return self.encoder.reparametrize(mean, log_var)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector into blendshape coefficients

        Parameters
        ----------
        latent : torch.Tensor
            (Batch_size, z_dim), Latent vectors (sampled)

        Returns
        -------
        torch.Tensor
            (Batch_size, sample_seq_len, channels), Blendshape coefficients
        """
        return self.decoder(latent)
