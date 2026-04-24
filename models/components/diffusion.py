from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Sinusoidal embedding for continuous EDM noise levels."""

    def __init__(self, num_channels: int, max_positions: int = 10000) -> None:
        super().__init__()
        if int(num_channels) % 2 != 0:
            raise ValueError("num_channels must be even for sinusoidal embeddings.")
        self.num_channels = int(num_channels)
        self.max_positions = int(max_positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            dtype=torch.float32,
            device=x.device,
        )
        freqs = freqs / max(self.num_channels // 2, 1)
        freqs = (1.0 / float(self.max_positions)) ** freqs
        angles = x.to(torch.float32).reshape(-1).ger(freqs.to(x.dtype))
        return torch.cat([angles.cos(), angles.sin()], dim=1)


class EmbeddingDenoiser(nn.Module):
    """MLP denoiser adapted from DiffGAD for node/target embeddings."""

    def __init__(
        self,
        embedding_dim: int,
        diffusion_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(embedding_dim), int(diffusion_dim))
        self.prototype_proj = nn.Linear(int(embedding_dim), int(diffusion_dim), bias=False)
        self.noise_embedding = PositionalEmbedding(int(diffusion_dim))
        self.noise_mlp = nn.Sequential(
            nn.Linear(int(diffusion_dim), int(diffusion_dim)),
            nn.SiLU(),
            nn.Linear(int(diffusion_dim), int(diffusion_dim)),
        )
        self.denoise_mlp = nn.Sequential(
            nn.Linear(int(diffusion_dim), int(diffusion_dim) * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(diffusion_dim) * 2, int(diffusion_dim) * 2),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(diffusion_dim) * 2, int(diffusion_dim)),
            nn.SiLU(),
            nn.Linear(int(diffusion_dim), int(embedding_dim)),
        )

    def forward(
        self,
        x: torch.Tensor,
        noise_level: torch.Tensor,
        prototype: torch.Tensor | None = None,
        prototype_alpha: float = 0.0,
    ) -> torch.Tensor:
        noise_emb = self.noise_mlp(self.noise_embedding(noise_level))
        hidden = self.input_proj(x) + noise_emb
        if prototype is not None and float(prototype_alpha) != 0.0:
            proto = prototype.to(dtype=x.dtype, device=x.device)
            if proto.dim() == 1:
                proto = proto.reshape(1, -1).expand(x.shape[0], -1)
            elif proto.shape[0] == 1:
                proto = proto.expand(x.shape[0], -1)
            if proto.shape != x.shape:
                raise ValueError(
                    f"prototype must broadcast to embedding shape {tuple(x.shape)}, got {tuple(proto.shape)}"
                )
            hidden = hidden + float(prototype_alpha) * self.prototype_proj(proto)
        return self.denoise_mlp(hidden)


class EDMPreconditioner(nn.Module):
    """EDM preconditioning used by DiffGAD-style embedding denoising."""

    def __init__(
        self,
        denoiser: EmbeddingDenoiser,
        sigma_data: float,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.sigma_data = float(sigma_data)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        prototype: torch.Tensor | None = None,
        prototype_alpha: float = 0.0,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        sigma_data_sq = self.sigma_data**2
        sigma_sq = sigma**2
        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma * self.sigma_data / (sigma_sq + sigma_data_sq).sqrt()
        c_in = 1.0 / (sigma_sq + sigma_data_sq).sqrt()
        c_noise = sigma.log().flatten() / 4.0
        residual = self.denoiser(
            c_in * x,
            c_noise,
            prototype=prototype,
            prototype_alpha=prototype_alpha,
        )
        return c_skip * x + c_out * residual.to(torch.float32)


class EmbeddingDiffusionRegularizer(nn.Module):
    """Diffusion denoising objective for DyRIFT-GNN target embeddings.

    This keeps the reference DiffGAD idea, learning to denoise graph embeddings,
    but uses it as a lightweight auxiliary loss inside the dynamic fraud GNN.
    """

    def __init__(
        self,
        embedding_dim: int,
        diffusion_dim: int | None = None,
        dropout: float = 0.0,
        p_mean: float = -1.2,
        p_std: float = 1.2,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        min_batch_size: int = 8,
    ) -> None:
        super().__init__()
        resolved_dim = (
            max(int(embedding_dim) * 2, 32)
            if diffusion_dim is None or int(diffusion_dim) <= 0
            else int(diffusion_dim)
        )
        if resolved_dim % 2 != 0:
            resolved_dim += 1
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.sigma_data = float(sigma_data)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.min_batch_size = int(min_batch_size)
        self.preconditioner = EDMPreconditioner(
            denoiser=EmbeddingDenoiser(
                embedding_dim=int(embedding_dim),
                diffusion_dim=resolved_dim,
                dropout=float(dropout),
            ),
            sigma_data=self.sigma_data,
        )

    def forward(
        self,
        embedding: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
        train_mask: torch.Tensor | None = None,
        prototype: torch.Tensor | None = None,
        prototype_alpha: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if embedding.dim() != 2:
            raise ValueError("Embedding diffusion expects a 2D embedding tensor.")
        if train_mask is not None:
            selected_mask = train_mask.to(dtype=torch.bool, device=embedding.device).reshape(-1)
            if selected_mask.shape[0] != embedding.shape[0]:
                raise ValueError("train_mask must match the embedding batch size.")
            embedding = embedding[selected_mask]
            if sample_weight is not None:
                sample_weight = sample_weight.reshape(-1)[selected_mask]
        if int(embedding.shape[0]) < max(self.min_batch_size, 1):
            return embedding.new_tensor(0.0), {
                "diffusion_selected_count": float(embedding.shape[0]),
                "diffusion_reconstruction_score": 0.0,
                "diffusion_sigma_mean": 0.0,
            }

        target = F.layer_norm(embedding, (embedding.shape[-1],))
        proto = None
        if prototype is not None and float(prototype_alpha) != 0.0:
            proto = F.layer_norm(
                prototype.to(dtype=target.dtype, device=target.device).reshape(1, -1),
                (target.shape[-1],),
            )
        rnd_normal = torch.randn(target.shape[0], device=target.device, dtype=target.dtype)
        sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        noisy = target + torch.randn_like(target) * sigma.reshape(-1, 1)
        reconstructed = self.preconditioner(
            noisy,
            sigma,
            prototype=proto,
            prototype_alpha=float(prototype_alpha),
        )

        reconstruction_error = (reconstructed - target).pow(2)
        per_sample_mse = reconstruction_error.mean(dim=-1)
        edm_weight = (sigma.pow(2) + self.sigma_data**2) / (sigma * self.sigma_data).pow(2)
        loss = edm_weight * per_sample_mse
        if sample_weight is not None:
            normalized_weight = sample_weight.to(dtype=loss.dtype, device=loss.device).reshape(-1)
            if normalized_weight.shape[0] != loss.shape[0]:
                raise ValueError("sample_weight must match the embedding batch size.")
            loss = loss * (normalized_weight / normalized_weight.mean().clamp_min(1e-6))

        score = reconstruction_error.sum(dim=-1).sqrt()
        metrics = {
            "diffusion_selected_count": float(target.shape[0]),
            "diffusion_reconstruction_score": float(score.detach().mean().item()),
            "diffusion_sigma_mean": float(sigma.detach().mean().item()),
        }
        if proto is not None:
            metrics["diffusion_proto_norm"] = float(proto.detach().norm(dim=-1).mean().item())
        return loss.mean(), metrics

    def reconstruction_score(
        self,
        embedding: torch.Tensor,
        sigma: float | None = None,
        prototype: torch.Tensor | None = None,
        prototype_alpha: float = 0.0,
    ) -> torch.Tensor:
        if embedding.dim() != 2:
            raise ValueError("Embedding diffusion expects a 2D embedding tensor.")
        target = F.layer_norm(embedding, (embedding.shape[-1],))
        sigma_value = self.sigma_data if sigma is None else float(sigma)
        sigma_tensor = target.new_full(
            (target.shape[0],),
            float(max(sigma_value, self.sigma_min)),
        )
        proto = None
        if prototype is not None and float(prototype_alpha) != 0.0:
            proto = F.layer_norm(
                prototype.to(dtype=target.dtype, device=target.device).reshape(1, -1),
                (target.shape[-1],),
            )
        reconstructed = self.preconditioner(
            target,
            sigma_tensor,
            prototype=proto,
            prototype_alpha=float(prototype_alpha),
        )
        return (reconstructed - target).pow(2).sum(dim=-1).sqrt()

    def sample_reconstruction(
        self,
        embedding: torch.Tensor,
        sigma: float | None = None,
        prototype: torch.Tensor | None = None,
        prototype_alpha: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embedding.dim() != 2:
            raise ValueError("Embedding diffusion expects a 2D embedding tensor.")
        target = F.layer_norm(embedding, (embedding.shape[-1],))
        sigma_value = self.sigma_data if sigma is None else float(sigma)
        sigma_tensor = target.new_full(
            (target.shape[0],),
            float(max(sigma_value, self.sigma_min)),
        )
        noisy = target + torch.randn_like(target) * sigma_tensor.reshape(-1, 1)
        proto = None
        if prototype is not None and float(prototype_alpha) != 0.0:
            proto = F.layer_norm(
                prototype.to(dtype=target.dtype, device=target.device).reshape(1, -1),
                (target.shape[-1],),
            )
        reconstructed = self.preconditioner(
            noisy,
            sigma_tensor,
            prototype=proto,
            prototype_alpha=float(prototype_alpha),
        )
        return target, reconstructed
