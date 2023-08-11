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
        super().__init__(
            audio_config=audio_config,
            audio_processor=audio_processor,
            noise_scheduler=noise_scheduler,
            in_channels=vae_z_dim,
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
        super().__init__(
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
