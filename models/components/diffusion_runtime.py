from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from models.components.diffusion import EmbeddingDiffusionRegularizer


class EmbeddingDiffusionRuntime:
    """Runtime glue for embedding diffusion training and inference hooks."""

    def __init__(
        self,
        *,
        config: Any,
        hidden_dim: int,
        device: torch.device,
    ) -> None:
        self.config = config
        self.hidden_dim = int(hidden_dim)
        self.device = device
        self._score_mean_ema: float | None = None
        self._score_std_ema: float | None = None
        self._prototype_ema: torch.Tensor | None = None

    def enabled(self) -> bool:
        return (
            float(self.config.embedding_diffusion_weight) > 0.0
            or float(self.config.embedding_diffusion_score_loss_weight) > 0.0
            or float(self.config.embedding_diffusion_view_weight) > 0.0
            or float(self.config.embedding_diffusion_inference_score_weight) != 0.0
        )

    def denoising_enabled(self) -> bool:
        return self.enabled() and float(self.config.embedding_diffusion_weight) > 0.0

    def score_loss_enabled(self) -> bool:
        return self.enabled() and float(self.config.embedding_diffusion_score_loss_weight) > 0.0

    def view_loss_enabled(self) -> bool:
        return self.enabled() and float(self.config.embedding_diffusion_view_weight) > 0.0

    def prototype_enabled(self) -> bool:
        mode = str(getattr(self.config, "embedding_diffusion_proto_mode", "none")).strip().lower()
        return (
            self.enabled()
            and mode not in {"", "none", "off", "false"}
            and float(getattr(self.config, "embedding_diffusion_proto_alpha", 0.0)) != 0.0
        )

    def detach_enabled(self) -> bool:
        return self.enabled() and bool(getattr(self.config, "embedding_diffusion_detach", False))

    def prediction_enabled(
        self,
        regularizer: EmbeddingDiffusionRegularizer | None,
        ready: bool = True,
        epoch: int | None = None,
    ) -> bool:
        inference_start_epoch = max(
            int(getattr(self.config, "embedding_diffusion_inference_start_epoch", 1)),
            1,
        )
        return (
            self.enabled()
            and bool(ready)
            and (epoch is None or int(epoch) >= inference_start_epoch)
            and float(self.config.embedding_diffusion_inference_score_weight) != 0.0
            and regularizer is not None
        )

    def start_epoch(self) -> int:
        return max(int(self.config.embedding_diffusion_start_epoch), 1)

    def reset_score_calibration(self) -> None:
        self._score_mean_ema = None
        self._score_std_ema = None

    def reset_prototype(self) -> None:
        self._prototype_ema = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "score_mean_ema": self._score_mean_ema,
            "score_std_ema": self._score_std_ema,
            "prototype_ema": (
                None
                if self._prototype_ema is None
                else self._prototype_ema.detach().cpu()
            ),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        if not state:
            self._score_mean_ema = None
            self._score_std_ema = None
            self._prototype_ema = None
            return
        self._score_mean_ema = (
            None if state.get("score_mean_ema") is None else float(state["score_mean_ema"])
        )
        self._score_std_ema = (
            None if state.get("score_std_ema") is None else float(state["score_std_ema"])
        )
        prototype = state.get("prototype_ema")
        self._prototype_ema = (
            None
            if prototype is None
            else prototype.to(device=self.device, dtype=torch.float32).detach().clone()
        )

    def build_regularizer(self) -> EmbeddingDiffusionRegularizer | None:
        if not self.enabled():
            return None
        return EmbeddingDiffusionRegularizer(
            embedding_dim=int(self.hidden_dim),
            diffusion_dim=int(self.config.embedding_diffusion_dim),
            dropout=float(self.config.dropout),
            p_mean=float(self.config.embedding_diffusion_p_mean),
            p_std=float(self.config.embedding_diffusion_p_std),
            sigma_data=float(self.config.embedding_diffusion_sigma_data),
            min_batch_size=int(self.config.embedding_diffusion_min_batch_size),
        ).to(self.device)

    def build_train_mask(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor | None:
        target = str(self.config.embedding_diffusion_target).strip().lower()
        label_vec = labels.reshape(-1)
        if target in {"", "all"}:
            return None
        if target in {"normal", "normal_only"}:
            return label_vec <= 0.5
        if target in {"fraud", "fraud_only", "anomaly"}:
            return label_vec > 0.5
        if target in {"labeled", "binary"}:
            label_code = label_vec.to(dtype=torch.long)
            return (label_code == 0) | (label_code == 1)
        raise ValueError(f"Unsupported embedding diffusion target: {self.config.embedding_diffusion_target}")

    def prototype_alpha(self) -> float:
        return float(getattr(self.config, "embedding_diffusion_proto_alpha", 0.0))

    def prototype_condition(
        self,
        *,
        embedding: torch.Tensor,
        train_mask: torch.Tensor | None,
        update: bool,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not self.prototype_enabled():
            return None, {}

        diagnostics: dict[str, float] = {}
        selected = embedding.detach()
        if train_mask is not None:
            selected_mask = train_mask.to(dtype=torch.bool, device=embedding.device).reshape(-1)
            if selected_mask.shape[0] != embedding.shape[0]:
                raise ValueError("train_mask must match the embedding batch size.")
            selected = selected[selected_mask]

        min_count = max(int(getattr(self.config, "embedding_diffusion_proto_min_count", 16)), 1)
        diagnostics["diffusion_proto_count"] = float(selected.shape[0])
        if update and selected.shape[0] >= min_count:
            with torch.no_grad():
                batch_proto = F.layer_norm(selected, (selected.shape[-1],)).mean(dim=0)
                if self._prototype_ema is None:
                    self._prototype_ema = batch_proto.detach().clone()
                else:
                    momentum = float(getattr(self.config, "embedding_diffusion_proto_momentum", 0.05))
                    momentum = min(1.0, max(0.0, momentum))
                    self._prototype_ema = (
                        (1.0 - momentum) * self._prototype_ema.to(batch_proto.device)
                        + momentum * batch_proto
                    ).detach()

        if self._prototype_ema is None:
            return None, diagnostics
        prototype = self._prototype_ema.to(device=embedding.device, dtype=embedding.dtype)
        diagnostics["diffusion_proto_norm"] = float(prototype.detach().norm().item())
        return prototype, diagnostics

    def _select_training_view(
        self,
        *,
        embedding: torch.Tensor,
        sample_weight: torch.Tensor | None,
        train_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        selected_embedding = embedding
        selected_weight = sample_weight
        if train_mask is not None:
            selected_mask = train_mask.to(dtype=torch.bool, device=embedding.device).reshape(-1)
            if selected_mask.shape[0] != embedding.shape[0]:
                raise ValueError("train_mask must match the embedding batch size.")
            selected_embedding = embedding[selected_mask]
            if sample_weight is not None:
                selected_weight = sample_weight.reshape(-1)[selected_mask]

        sample_size = max(int(self.config.embedding_diffusion_view_sample_size), 0)
        if sample_size > 0 and selected_embedding.shape[0] > sample_size:
            order = torch.randperm(
                selected_embedding.shape[0],
                device=selected_embedding.device,
            )[:sample_size]
            selected_embedding = selected_embedding[order]
            if selected_weight is not None:
                selected_weight = selected_weight.reshape(-1)[order]
        return selected_embedding, selected_weight

    def compute_score_loss(
        self,
        *,
        regularizer: EmbeddingDiffusionRegularizer,
        embedding: torch.Tensor,
        targets: torch.Tensor,
        sample_weight: torch.Tensor | None,
        primary_multiclass_enabled: bool,
        prototype: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.score_loss_enabled() or primary_multiclass_enabled:
            return embedding.new_tensor(0.0), {}
        if embedding.shape[0] < 4 or torch.unique(targets.detach()).numel() < 2:
            return embedding.new_tensor(0.0), {}

        score = regularizer.reconstruction_score(
            embedding,
            sigma=float(self.config.embedding_diffusion_score_sigma),
            prototype=prototype,
            prototype_alpha=self.prototype_alpha(),
        )
        self._update_score_calibration(score.detach())
        score_std = score.std(unbiased=False).clamp_min(1e-6)
        score_z = (score - score.mean()) / score_std
        score_z = float(self.config.embedding_diffusion_score_sign) * score_z
        logits = float(self.config.embedding_diffusion_score_temperature) * score_z
        loss_values = F.binary_cross_entropy_with_logits(
            logits,
            targets.to(dtype=logits.dtype),
            reduction="none",
        )
        if sample_weight is not None:
            normalized_weight = sample_weight.to(
                dtype=loss_values.dtype,
                device=loss_values.device,
            ).reshape(-1)
            loss_values = loss_values * (normalized_weight / normalized_weight.mean().clamp_min(1e-6))

        fraud_mask = targets > 0.5
        normal_mask = ~fraud_mask
        diagnostics = {
            "diffusion_score_loss": float(loss_values.detach().mean().item()),
            "diffusion_score_mean": float(score.detach().mean().item()),
            "diffusion_score_std": float(score_std.detach().item()),
        }
        if self._score_mean_ema is not None and self._score_std_ema is not None:
            diagnostics["diffusion_score_calibration_mean"] = float(self._score_mean_ema)
            diagnostics["diffusion_score_calibration_std"] = float(self._score_std_ema)
        if torch.any(fraud_mask):
            diagnostics["diffusion_score_fraud_mean"] = float(score[fraud_mask].detach().mean().item())
        if torch.any(normal_mask):
            diagnostics["diffusion_score_normal_mean"] = float(score[normal_mask].detach().mean().item())
        return loss_values.mean(), diagnostics

    def _update_score_calibration(self, score: torch.Tensor) -> None:
        if score.numel() < 2:
            return
        momentum = float(getattr(self.config, "embedding_diffusion_score_calibration_momentum", 0.1))
        momentum = min(1.0, max(0.0, momentum))
        batch_mean = float(score.mean().item())
        batch_std = float(score.std(unbiased=False).clamp_min(1e-6).item())
        if self._score_mean_ema is None or self._score_std_ema is None:
            self._score_mean_ema = batch_mean
            self._score_std_ema = batch_std
            return
        self._score_mean_ema = (1.0 - momentum) * self._score_mean_ema + momentum * batch_mean
        self._score_std_ema = (1.0 - momentum) * self._score_std_ema + momentum * batch_std

    def _normalize_score(self, score: torch.Tensor) -> torch.Tensor:
        calibration = str(
            getattr(self.config, "embedding_diffusion_score_calibration", "batch")
        ).strip().lower()
        if calibration in {"ema", "train_ema"} and self._score_mean_ema is not None:
            mean = score.new_tensor(float(self._score_mean_ema))
            std = score.new_tensor(float(max(self._score_std_ema or 0.0, 1e-6)))
            return (score - mean) / std
        return (score - score.mean()) / score.std(unbiased=False).clamp_min(1e-6)

    def compute_view_loss(
        self,
        *,
        regularizer: EmbeddingDiffusionRegularizer,
        embedding: torch.Tensor,
        sample_weight: torch.Tensor | None,
        train_mask: torch.Tensor | None,
        prototype: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not self.view_loss_enabled():
            return embedding.new_tensor(0.0), {}

        selected_embedding, selected_weight = self._select_training_view(
            embedding=embedding,
            sample_weight=sample_weight,
            train_mask=train_mask,
        )
        if selected_embedding.shape[0] < max(int(self.config.embedding_diffusion_min_batch_size), 4):
            return embedding.new_tensor(0.0), {
                "diffusion_view_count": float(selected_embedding.shape[0]),
                "diffusion_view_loss": 0.0,
            }

        sigma = float(self.config.embedding_diffusion_view_sigma)
        target, reconstructed = regularizer.sample_reconstruction(
            selected_embedding,
            sigma=sigma if sigma > 0.0 else None,
            prototype=prototype,
            prototype_alpha=self.prototype_alpha(),
        )
        z_target = F.normalize(target, dim=-1)
        z_view = F.normalize(reconstructed, dim=-1)
        mode = str(self.config.embedding_diffusion_view_mode).strip().lower()
        if mode not in {"infonce", "mse", "both"}:
            raise ValueError(f"Unsupported embedding diffusion view mode: {self.config.embedding_diffusion_view_mode}")

        sample_loss = selected_embedding.new_tensor(0.0)
        similarity = z_target @ z_view.T
        if mode in {"infonce", "both"}:
            temperature = max(float(self.config.embedding_diffusion_view_temperature), 1e-3)
            logits = similarity / temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            row_loss = F.cross_entropy(logits, labels, reduction="none")
            col_loss = F.cross_entropy(logits.T, labels, reduction="none")
            if selected_weight is not None:
                normalized_weight = selected_weight.to(
                    dtype=row_loss.dtype,
                    device=row_loss.device,
                ).reshape(-1)
                normalized_weight = normalized_weight / normalized_weight.mean().clamp_min(1e-6)
                row_loss = row_loss * normalized_weight
                col_loss = col_loss * normalized_weight
            sample_loss = sample_loss + 0.5 * (row_loss.mean() + col_loss.mean())
        if mode in {"mse", "both"}:
            mse_loss = (z_target - z_view).pow(2).mean(dim=-1)
            if selected_weight is not None:
                normalized_weight = selected_weight.to(
                    dtype=mse_loss.dtype,
                    device=mse_loss.device,
                ).reshape(-1)
                mse_loss = mse_loss * (normalized_weight / normalized_weight.mean().clamp_min(1e-6))
            sample_loss = sample_loss + mse_loss.mean()

        positive_similarity = similarity.diag()
        if similarity.numel() > positive_similarity.numel():
            negative_similarity = (
                similarity.sum() - positive_similarity.sum()
            ) / float(similarity.numel() - positive_similarity.numel())
        else:
            negative_similarity = selected_embedding.new_tensor(0.0)
        diagnostics = {
            "diffusion_view_count": float(selected_embedding.shape[0]),
            "diffusion_view_loss": float(sample_loss.detach().item()),
            "diffusion_view_pos_sim": float(positive_similarity.detach().mean().item()),
            "diffusion_view_neg_sim": float(negative_similarity.detach().item()),
        }
        return sample_loss, diagnostics

    def blend_probability(
        self,
        *,
        regularizer: EmbeddingDiffusionRegularizer | None,
        primary_probability: torch.Tensor,
        embedding: torch.Tensor | None,
        prototype: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            regularizer is None
            or embedding is None
            or float(self.config.embedding_diffusion_inference_score_weight) == 0.0
        ):
            return primary_probability
        score = regularizer.reconstruction_score(
            embedding,
            sigma=float(self.config.embedding_diffusion_score_sigma),
            prototype=prototype,
            prototype_alpha=self.prototype_alpha(),
        )
        if score.numel() < 2:
            return primary_probability

        score_z = self._normalize_score(score)
        score_z = float(self.config.embedding_diffusion_score_sign) * score_z
        diffusion_probability = torch.sigmoid(float(self.config.embedding_diffusion_score_temperature) * score_z)
        blend = min(1.0, max(-1.0, float(self.config.embedding_diffusion_inference_score_weight)))
        if blend >= 0.0:
            return (1.0 - blend) * primary_probability + blend * diffusion_probability
        inverse_blend = abs(blend)
        return (1.0 + blend) * primary_probability + inverse_blend * (1.0 - diffusion_probability)
