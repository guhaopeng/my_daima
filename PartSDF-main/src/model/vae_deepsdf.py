import torch
import torch.nn as nn
import torch.nn.functional as F

from .deepsdf import DeepSDF, LatentModulated, LatentDemodulated


class PointNetEncoder(nn.Module):
    """Lightweight PointNet-style encoder for global shape features."""

    def __init__(
        self,
        input_dim=3,
        hidden_dims=(64, 128, 256),
        global_dim=1024,
        use_batchnorm=True,
    ):
        super().__init__()
        dims = (input_dim,) + tuple(hidden_dims) + (global_dim,)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_batchnorm = use_batchnorm

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1))
            if use_batchnorm:
                self.norms.append(nn.BatchNorm1d(out_dim))

    def forward(self, surface_points: torch.Tensor) -> torch.Tensor:
        if surface_points.ndim != 3:
            raise ValueError(
                "surface_points must have shape (B, N, 3) or (B, 3, N)."
            )
        if surface_points.shape[-1] == 3:
            x = surface_points.transpose(1, 2)
        elif surface_points.shape[1] == 3:
            x = surface_points
        else:
            raise ValueError(
                "surface_points must have 3 channels in the last or second dimension."
            )

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batchnorm:
                x = self.norms[i](x)
            if i < len(self.layers) - 1:
                x = F.relu(x, inplace=True)

        return torch.max(x, dim=-1).values


class VAEDeepSDF(nn.Module):
    """Point-cloud encoder + VAE bottleneck + DeepSDF-style decoder."""

    def __init__(
        self,
        latent_dim=64,
        encoder_global_dim=1024,
        encoder_hidden_dims=(64, 128, 256),
        encoder_use_batchnorm=True,
        decoder_type="latentmodulated",
        decoder_specs=None,
        use_occ=False,
        **kwargs,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_occ = use_occ

        self.encoder = PointNetEncoder(
            input_dim=3,
            hidden_dims=encoder_hidden_dims,
            global_dim=encoder_global_dim,
            use_batchnorm=encoder_use_batchnorm,
        )
        self.fc_mu = nn.Linear(encoder_global_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_global_dim, latent_dim)

        decoder_specs = {} if decoder_specs is None else dict(decoder_specs)
        decoder_specs.setdefault("latent_dim", latent_dim)
        self.decoder = self._build_decoder(decoder_type, decoder_specs)

    @staticmethod
    def _build_decoder(decoder_type: str, decoder_specs: dict) -> nn.Module:
        decoder_type = decoder_type.lower()
        if decoder_type == "deepsdf":
            return DeepSDF(**decoder_specs)
        if decoder_type == "latentmodulated":
            return LatentModulated(**decoder_specs)
        if decoder_type == "latentdemodulated":
            return LatentDemodulated(**decoder_specs)
        raise NotImplementedError(f'Unknown decoder type "{decoder_type}".')

    def encode_distribution(
        self, surface_points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoder(surface_points)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_surface(
        self,
        surface_points: torch.Tensor,
        sample: bool = True,
        return_dist: bool = False,
    ):
        mu, logvar = self.encode_distribution(surface_points)
        latent = self.reparameterize(mu, logvar) if sample else mu
        if return_dist:
            return latent, mu, logvar
        return latent

    def decode(self, latent: torch.Tensor, xyz: torch.Tensor, **kwargs) -> torch.Tensor:
        if latent.ndim == 2:
            latent = latent.unsqueeze(1)
        return self.decoder(latent, xyz, **kwargs)

    def forward(self, latent: torch.Tensor, xyz: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.decode(latent, xyz, **kwargs)

    def forward_with_surface(
        self,
        surface_points: torch.Tensor,
        xyz: torch.Tensor,
        sample: bool = True,
        return_dist: bool = False,
        **kwargs,
    ):
        latent, mu, logvar = self.encode_surface(
            surface_points, sample=sample, return_dist=True
        )
        sdf = self.decode(latent, xyz, **kwargs)
        if return_dist:
            return sdf, latent, mu, logvar
        return sdf
