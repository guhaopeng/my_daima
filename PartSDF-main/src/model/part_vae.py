from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfacePointEncoder(nn.Module):
    """Self-contained PointNet-style encoder to avoid cross-file version mismatch."""

    def __init__(
        self,
        in_dim: int = 3,
        encoder_channels: list[int] | None = None,
        encoder_fc_dim: int = 256,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512]

        layers = []
        prev_dim = in_dim
        for out_dim in encoder_channels:
            layers.append(nn.Conv1d(prev_dim, out_dim, kernel_size=1))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(prev_dim, encoder_fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_fc_dim, encoder_fc_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, surface_points: torch.Tensor) -> torch.Tensor:
        if surface_points.ndim != 3:
            raise ValueError("surface_points must have shape [B, N, 3].")
        feats = self.mlp(surface_points.transpose(1, 2))
        feats = torch.max(feats, dim=-1).values
        feats = self.fc(feats)
        return feats


class PartAwareVAE(nn.Module):
    """Hierarchical Part-aware VAE that predicts part latents and part poses."""

    def __init__(
        self,
        n_parts: int,
        part_latent_dim: int,
        pose_dim: int = 10,
        global_latent_dim: int = 64,
        part_latent_code_dim: int = 16,
        encoder_channels: list[int] | None = None,
        encoder_fc_dim: int = 256,
        part_id_dim: int = 16,
        token_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff_dim: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_parts = n_parts
        self.part_latent_dim = part_latent_dim
        self.pose_dim = pose_dim
        self.global_latent_dim = global_latent_dim
        self.part_latent_code_dim = part_latent_code_dim
        self.total_latent_dim = global_latent_dim + n_parts * part_latent_code_dim

        self.surface_encoder = SurfacePointEncoder(
            in_dim=3,
            encoder_channels=encoder_channels,
            encoder_fc_dim=encoder_fc_dim,
        )
        self.part_id_embedding = nn.Embedding(n_parts, part_id_dim)
        self.part_token_mlp = nn.Sequential(
            nn.Linear(encoder_fc_dim + pose_dim + part_id_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.part_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.global_mu = nn.Linear(token_dim, global_latent_dim)
        self.global_logvar = nn.Linear(token_dim, global_latent_dim)
        self.part_mu = nn.Linear(token_dim, part_latent_code_dim)
        self.part_logvar = nn.Linear(token_dim, part_latent_code_dim)

        decoder_in_dim = global_latent_dim + part_latent_code_dim + part_id_dim
        self.part_decoder = nn.Sequential(
            nn.Linear(decoder_in_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
        )
        self.part_latent_head = nn.Linear(token_dim, part_latent_dim)
        self.pose_head = nn.Linear(token_dim, pose_dim)

    def _safe_standardize_quaternion(self, quat: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        safe_quat = quat / norm.clamp_min(1e-8)
        default_quat = torch.zeros_like(safe_quat)
        default_quat[..., 0] = 1.0
        safe_quat = torch.where(norm > 1e-8, safe_quat, default_quat)
        return torch.where(safe_quat[..., 0:1] < 0, -safe_quat, safe_quat)

    def _build_part_tokens(
        self, surface_points: torch.Tensor, part_pose: torch.Tensor
    ) -> torch.Tensor:
        if surface_points.ndim != 3:
            raise ValueError("surface_points must have shape [B, N, 3].")
        if part_pose.ndim != 3:
            raise ValueError("part_pose must have shape [B, P, K].")
        if part_pose.shape[1] != self.n_parts:
            raise ValueError(f"Expected {self.n_parts} parts, got {part_pose.shape[1]}.")
        if part_pose.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected pose_dim={self.pose_dim}, got {part_pose.shape[-1]}.")

        batch_size = surface_points.shape[0]
        global_feat = self.surface_encoder(surface_points)
        global_feat = global_feat[:, None, :].expand(-1, self.n_parts, -1)

        part_ids = torch.arange(self.n_parts, device=surface_points.device)
        part_ids = self.part_id_embedding(part_ids)[None].expand(batch_size, -1, -1)

        tokens = torch.cat([global_feat, part_pose, part_ids], dim=-1)
        tokens = self.part_token_mlp(tokens)
        tokens = self.part_transformer(tokens)
        return tokens

    def encode_distribution(
        self, surface_points: torch.Tensor, part_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self._build_part_tokens(surface_points, part_pose)
        pooled = tokens.mean(dim=1)
        mu_g = self.global_mu(pooled)
        logvar_g = self.global_logvar(pooled)
        mu_p = self.part_mu(tokens)
        logvar_p = self.part_logvar(tokens)
        return mu_g, logvar_g, mu_p, logvar_p

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(
        self,
        surface_points: torch.Tensor,
        part_pose: torch.Tensor,
        sample: bool = True,
        return_dist: bool = False,
    ):
        mu_g, logvar_g, mu_p, logvar_p = self.encode_distribution(surface_points, part_pose)
        z_g = self.reparameterize(mu_g, logvar_g) if sample else mu_g
        z_p = self.reparameterize(mu_p, logvar_p) if sample else mu_p
        if return_dist:
            return z_g, z_p, mu_g, logvar_g, mu_p, logvar_p
        return z_g, z_p

    def decode(self, z_g: torch.Tensor, z_p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z_g.ndim != 2:
            raise ValueError("z_g must have shape [B, Dg].")
        if z_p.ndim != 3:
            raise ValueError("z_p must have shape [B, P, Dp].")
        if z_p.shape[1] != self.n_parts:
            raise ValueError(f"Expected {self.n_parts} parts, got {z_p.shape[1]}.")

        batch_size = z_g.shape[0]
        z_g_expand = z_g[:, None, :].expand(-1, self.n_parts, -1)
        part_ids = torch.arange(self.n_parts, device=z_g.device)
        part_ids = self.part_id_embedding(part_ids)[None].expand(batch_size, -1, -1)
        decoder_input = torch.cat([z_g_expand, z_p, part_ids], dim=-1)

        part_features = self.part_decoder(decoder_input)
        pred_part_latent = self.part_latent_head(part_features)
        pred_pose = self.pose_head(part_features)
        pred_pose = self._project_pose(pred_pose)
        return pred_part_latent, pred_pose

    def _project_pose(self, pose: torch.Tensor) -> torch.Tensor:
        if self.pose_dim < 4:
            return pose

        quat = self._safe_standardize_quaternion(pose[..., :4])
        if self.pose_dim <= 7:
            tail = pose[..., 4:]
            return torch.cat([quat, tail], dim=-1)

        trans = pose[..., 4:7]
        scales = F.softplus(pose[..., 7:]) + 1e-3
        return torch.cat([quat, trans, scales], dim=-1)

    def forward(
        self,
        surface_points: torch.Tensor,
        part_pose: torch.Tensor,
        sample: bool = True,
        return_dist: bool = False,
    ):
        out = self.encode(surface_points, part_pose, sample=sample, return_dist=True)
        z_g, z_p, mu_g, logvar_g, mu_p, logvar_p = out
        pred_part_latent, pred_pose = self.decode(z_g, z_p)
        if return_dist:
            return pred_part_latent, pred_pose, mu_g, logvar_g, mu_p, logvar_p, z_g, z_p
        return pred_part_latent, pred_pose
