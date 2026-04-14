from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PrototypeMemoryConfig:
    num_classes: int
    embedding_dim: int
    temperature: float = 0.2
    momentum: float = 0.9
    start_epoch: int = 1
    loss_ramp_epochs: int = 0
    bucket_mode: str = "global"
    num_buckets: int = 1
    neighbor_blend: float = 0.0
    global_blend: float = 0.0
    consistency_weight: float = 0.0
    separation_weight: float = 0.0
    separation_margin: float = 0.1

    def __post_init__(self) -> None:
        if self.bucket_mode not in {"global", "time_bucket"}:
            raise ValueError(f"Unsupported prototype bucket mode: {self.bucket_mode}")

    @property
    def enabled(self) -> bool:
        return int(self.num_classes) >= 2

    @property
    def use_time_buckets(self) -> bool:
        return self.bucket_mode == "time_bucket" and int(self.num_buckets) > 1


@dataclass
class PrototypeMemoryState:
    global_memory: torch.Tensor
    global_initialized: torch.Tensor
    bucket_memory: torch.Tensor | None = None
    bucket_initialized: torch.Tensor | None = None


class PrototypeMemoryBank:
    def __init__(
        self,
        *,
        config: PrototypeMemoryConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.state = PrototypeMemoryState(
            global_memory=torch.zeros(
                (int(config.num_classes), int(config.embedding_dim)),
                dtype=torch.float32,
                device=device,
            ),
            global_initialized=torch.zeros(
                int(config.num_classes),
                dtype=torch.bool,
                device=device,
            ),
        )
        if self.config.use_time_buckets:
            self.state.bucket_memory = torch.zeros(
                (
                    int(config.num_buckets),
                    int(config.num_classes),
                    int(config.embedding_dim),
                ),
                dtype=torch.float32,
                device=device,
            )
            self.state.bucket_initialized = torch.zeros(
                (int(config.num_buckets), int(config.num_classes)),
                dtype=torch.bool,
                device=device,
            )
        self._last_metrics = self._empty_metrics(epoch=0, active=False)

    def active(self, epoch: int) -> bool:
        return self.config.enabled and int(epoch) >= max(int(self.config.start_epoch), 1)

    @property
    def last_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def _empty_metrics(self, *, epoch: int, active: bool) -> dict[str, float]:
        return {
            "prototype_active_epoch": 1.0 if active else 0.0,
            "prototype_loss_ramp": float(self._loss_ramp(epoch) if active else 0.0),
            "prototype_memory_initialized_ratio": float(
                self.state.global_initialized.to(dtype=torch.float32).mean().detach().item()
            )
            if self.state.global_initialized.numel()
            else 0.0,
            "prototype_target_available_ratio": 0.0,
            "prototype_classification_ratio": 0.0,
            "prototype_classification_loss": 0.0,
            "prototype_consistency_loss": 0.0,
            "prototype_separation_loss": 0.0,
            "prototype_margin": 0.0,
            "prototype_temporal_stability": 0.0,
        }

    def compute_loss(
        self,
        *,
        embedding: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
        sample_weight: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
        bucket_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active = self.active(epoch)
        metrics = self._empty_metrics(epoch=epoch, active=active)
        if not active:
            self._last_metrics = metrics
            return embedding.new_tensor(0.0)

        valid_mask = targets >= 0
        if not torch.any(valid_mask):
            self._last_metrics = metrics
            return embedding.new_tensor(0.0)

        valid_embedding = embedding[valid_mask]
        valid_targets = targets[valid_mask]
        valid_weight = sample_weight[valid_mask] if sample_weight is not None else None
        valid_bucket_ids = bucket_ids[valid_mask] if bucket_ids is not None else None

        normalized_embedding = F.normalize(valid_embedding, dim=-1)
        loss_state = self._snapshot_state()
        logits = self._build_logits(
            normalized_embedding=normalized_embedding,
            bucket_ids=valid_bucket_ids,
            state=loss_state,
        )
        target_available = logits["available"][
            torch.arange(valid_targets.shape[0], device=valid_targets.device),
            valid_targets,
        ]
        active_class_count = logits["available"].sum(dim=1)
        classification_mask = target_available & (active_class_count > 1)
        total_loss = embedding.new_tensor(0.0)
        metrics["prototype_target_available_ratio"] = float(
            target_available.to(dtype=torch.float32).mean().detach().item()
        )
        metrics["prototype_classification_ratio"] = float(
            classification_mask.to(dtype=torch.float32).mean().detach().item()
        )
        metrics["prototype_temporal_stability"] = self._compute_temporal_stability(
            targets=valid_targets,
            bucket_ids=valid_bucket_ids,
            state=loss_state,
        )

        if torch.any(classification_mask):
            classification_loss = F.cross_entropy(
                logits["logits"][classification_mask],
                valid_targets[classification_mask],
                reduction="none",
            )
            if class_weight is not None:
                classification_loss = classification_loss * class_weight[valid_targets[classification_mask]]
            if valid_weight is not None:
                classification_loss = classification_loss * valid_weight[classification_mask]
            metrics["prototype_classification_loss"] = float(classification_loss.mean().detach().item())
            total_loss = total_loss + classification_loss.mean()

        consistency_weight_value = float(np.clip(self.config.consistency_weight, 0.0, 10.0))
        if consistency_weight_value > 0.0:
            anchor = self._build_temporal_anchor(bucket_ids=valid_bucket_ids, state=loss_state)
            anchor_available = anchor["available"][
                torch.arange(valid_targets.shape[0], device=valid_targets.device),
                valid_targets,
            ]
            if torch.any(anchor_available):
                selected_embedding = normalized_embedding[anchor_available]
                selected_anchor = anchor["memory"][
                    anchor_available,
                    valid_targets[anchor_available],
                ]
                consistency_loss = 1.0 - torch.sum(selected_embedding * selected_anchor, dim=-1)
                if class_weight is not None:
                    consistency_loss = consistency_loss * class_weight[valid_targets[anchor_available]]
                if valid_weight is not None:
                    consistency_loss = consistency_loss * valid_weight[anchor_available]
                metrics["prototype_consistency_loss"] = float(consistency_loss.mean().detach().item())
                total_loss = total_loss + consistency_weight_value * consistency_loss.mean()

        separation_weight_value = float(np.clip(self.config.separation_weight, 0.0, 10.0))
        if separation_weight_value > 0.0 and torch.any(classification_mask):
            temperature = max(float(self.config.temperature), 1e-3)
            similarity_logits = logits["logits"][classification_mask] * temperature
            similarity_available = logits["available"][classification_mask]
            selected_targets = valid_targets[classification_mask]
            target_similarity = similarity_logits[
                torch.arange(selected_targets.shape[0], device=selected_targets.device),
                selected_targets,
            ]
            negative_mask = similarity_available.clone()
            negative_mask[
                torch.arange(selected_targets.shape[0], device=selected_targets.device),
                selected_targets,
            ] = False
            if torch.any(torch.any(negative_mask, dim=1)):
                hardest_negative = similarity_logits.masked_fill(~negative_mask, -1e4).max(dim=1).values
                valid_separation = torch.any(negative_mask, dim=1)
                separation_margin = (
                    target_similarity[valid_separation] - hardest_negative[valid_separation]
                )
                metrics["prototype_margin"] = float(separation_margin.mean().detach().item())
                separation_loss = F.relu(
                    float(self.config.separation_margin)
                    - separation_margin
                )
                if class_weight is not None:
                    separation_loss = separation_loss * class_weight[selected_targets[valid_separation]]
                if valid_weight is not None:
                    separation_loss = separation_loss * valid_weight[classification_mask][valid_separation]
                metrics["prototype_separation_loss"] = float(separation_loss.mean().detach().item())
                total_loss = total_loss + separation_weight_value * separation_loss.mean()

        self._update_memory(
            memory=self.state.global_memory,
            initialized=self.state.global_initialized,
            embedding=valid_embedding,
            targets=valid_targets,
        )
        if self.config.use_time_buckets:
            if valid_bucket_ids is None or self.state.bucket_memory is None or self.state.bucket_initialized is None:
                raise ValueError("Time-bucket prototype memory requires per-sample bucket ids.")
            self._update_bucket_memory(
                embedding=valid_embedding,
                targets=valid_targets,
                bucket_ids=valid_bucket_ids,
            )
        metrics["prototype_memory_initialized_ratio"] = float(
            self.state.global_initialized.to(dtype=torch.float32).mean().detach().item()
        )
        self._last_metrics = metrics

        if float(total_loss.detach().item()) <= 0.0:
            return embedding.new_tensor(0.0)
        return self._loss_ramp(epoch) * total_loss

    def _update_bucket_memory(
        self,
        *,
        embedding: torch.Tensor,
        targets: torch.Tensor,
        bucket_ids: torch.Tensor,
    ) -> None:
        if self.state.bucket_memory is None or self.state.bucket_initialized is None:
            return
        clamped_bucket_ids = bucket_ids.to(dtype=torch.long).clamp_(0, self.state.bucket_memory.shape[0] - 1)
        for bucket_value in torch.unique(clamped_bucket_ids).tolist():
            bucket_id = int(bucket_value)
            bucket_mask = clamped_bucket_ids == bucket_id
            if not torch.any(bucket_mask):
                continue
            self._update_memory(
                memory=self.state.bucket_memory[bucket_id],
                initialized=self.state.bucket_initialized[bucket_id],
                embedding=embedding[bucket_mask],
                targets=targets[bucket_mask],
            )

    def _update_memory(
        self,
        *,
        memory: torch.Tensor,
        initialized: torch.Tensor,
        embedding: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        momentum = float(np.clip(self.config.momentum, 0.0, 0.999))
        normalized_embedding = F.normalize(embedding.detach(), dim=-1)
        with torch.no_grad():
            for class_value in torch.unique(targets).tolist():
                class_id = int(class_value)
                class_mask = targets == class_id
                if not torch.any(class_mask):
                    continue
                class_mean = normalized_embedding[class_mask].mean(dim=0)
                class_mean = F.normalize(class_mean.unsqueeze(0), dim=-1).squeeze(0)
                if bool(initialized[class_id].item()):
                    blended = momentum * memory[class_id] + (1.0 - momentum) * class_mean
                    memory[class_id] = F.normalize(blended.unsqueeze(0), dim=-1).squeeze(0)
                else:
                    memory[class_id] = class_mean
                    initialized[class_id] = True

    def _loss_ramp(self, epoch: int) -> float:
        ramp_epochs = max(int(self.config.loss_ramp_epochs), 0)
        if ramp_epochs <= 1:
            return 1.0
        progress = int(epoch) - max(int(self.config.start_epoch), 1) + 1
        return float(np.clip(progress / ramp_epochs, 0.0, 1.0))

    def _snapshot_state(self) -> PrototypeMemoryState:
        bucket_memory = None
        bucket_initialized = None
        if self.state.bucket_memory is not None:
            bucket_memory = self.state.bucket_memory.detach().clone()
        if self.state.bucket_initialized is not None:
            bucket_initialized = self.state.bucket_initialized.clone()
        return PrototypeMemoryState(
            global_memory=self.state.global_memory.detach().clone(),
            global_initialized=self.state.global_initialized.clone(),
            bucket_memory=bucket_memory,
            bucket_initialized=bucket_initialized,
        )

    def _clamp_bucket_ids(self, bucket_ids: torch.Tensor, state: PrototypeMemoryState) -> torch.Tensor:
        if state.bucket_memory is None:
            raise ValueError("Time-bucket prototype memory requires initialized bucket memory.")
        return bucket_ids.to(dtype=torch.long).clamp_(0, state.bucket_memory.shape[0] - 1)

    def _normalize_memory_rows(
        self,
        *,
        memory: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.any(available):
            return memory
        normalized = memory.clone()
        flat_memory = normalized.reshape(-1, normalized.shape[-1])
        flat_available = available.reshape(-1)
        flat_memory[flat_available] = F.normalize(flat_memory[flat_available], dim=-1)
        return normalized

    def _gather_bucket_memory(
        self,
        *,
        bucket_ids: torch.Tensor | None,
        state: PrototypeMemoryState,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if bucket_ids is None or state.bucket_memory is None or state.bucket_initialized is None:
            return None, None
        clamped_bucket_ids = self._clamp_bucket_ids(bucket_ids, state)
        return state.bucket_memory[clamped_bucket_ids], state.bucket_initialized[clamped_bucket_ids]

    def _gather_neighbor_memory(
        self,
        *,
        bucket_ids: torch.Tensor | None,
        state: PrototypeMemoryState,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if bucket_ids is None or state.bucket_memory is None or state.bucket_initialized is None:
            return None, None
        clamped_bucket_ids = self._clamp_bucket_ids(bucket_ids, state)
        num_samples = clamped_bucket_ids.shape[0]
        num_classes = state.bucket_memory.shape[1]
        embedding_dim = state.bucket_memory.shape[2]
        accum = torch.zeros(
            (num_samples, num_classes, embedding_dim),
            dtype=state.bucket_memory.dtype,
            device=self.device,
        )
        count = torch.zeros(
            (num_samples, num_classes),
            dtype=state.bucket_memory.dtype,
            device=self.device,
        )

        has_prev = clamped_bucket_ids > 0
        if torch.any(has_prev):
            prev_ids = clamped_bucket_ids[has_prev] - 1
            accum[has_prev] = accum[has_prev] + state.bucket_memory[prev_ids]
            count[has_prev] = count[has_prev] + state.bucket_initialized[prev_ids].to(accum.dtype)

        has_next = clamped_bucket_ids < (state.bucket_memory.shape[0] - 1)
        if torch.any(has_next):
            next_ids = clamped_bucket_ids[has_next] + 1
            accum[has_next] = accum[has_next] + state.bucket_memory[next_ids]
            count[has_next] = count[has_next] + state.bucket_initialized[next_ids].to(accum.dtype)

        available = count > 0
        if not torch.any(available):
            return None, None
        neighbor_memory = accum / count.clamp_min(1.0).unsqueeze(-1)
        neighbor_memory = self._normalize_memory_rows(memory=neighbor_memory, available=available)
        return neighbor_memory, available

    def _build_temporal_anchor(
        self,
        *,
        bucket_ids: torch.Tensor | None,
        state: PrototypeMemoryState,
    ) -> dict[str, torch.Tensor]:
        num_classes = state.global_memory.shape[0]
        embedding_dim = state.global_memory.shape[1]
        num_samples = 1 if bucket_ids is None else int(bucket_ids.shape[0])
        anchor_memory = torch.zeros(
            (num_samples, num_classes, embedding_dim),
            dtype=state.global_memory.dtype,
            device=self.device,
        )
        anchor_count = torch.zeros(
            (num_samples, num_classes),
            dtype=state.global_memory.dtype,
            device=self.device,
        )

        current_memory, current_available = self._gather_bucket_memory(bucket_ids=bucket_ids, state=state)
        if current_memory is not None and current_available is not None:
            anchor_memory = anchor_memory + torch.where(
                current_available.unsqueeze(-1),
                current_memory,
                torch.zeros_like(current_memory),
            )
            anchor_count = anchor_count + current_available.to(anchor_count.dtype)

        neighbor_memory, neighbor_available = self._gather_neighbor_memory(bucket_ids=bucket_ids, state=state)
        if neighbor_memory is not None and neighbor_available is not None:
            anchor_memory = anchor_memory + torch.where(
                neighbor_available.unsqueeze(-1),
                neighbor_memory,
                torch.zeros_like(neighbor_memory),
            )
            anchor_count = anchor_count + neighbor_available.to(anchor_count.dtype)

        global_available = state.global_initialized.unsqueeze(0).expand(num_samples, -1)
        global_memory = state.global_memory.unsqueeze(0).expand(num_samples, -1, -1)
        anchor_memory = anchor_memory + torch.where(
            global_available.unsqueeze(-1),
            global_memory,
            torch.zeros_like(global_memory),
        )
        anchor_count = anchor_count + global_available.to(anchor_count.dtype)

        available = anchor_count > 0
        anchor_memory = anchor_memory / anchor_count.clamp_min(1.0).unsqueeze(-1)
        anchor_memory = self._normalize_memory_rows(memory=anchor_memory, available=available)
        return {"memory": anchor_memory, "available": available}

    def _compute_temporal_stability(
        self,
        *,
        targets: torch.Tensor,
        bucket_ids: torch.Tensor | None,
        state: PrototypeMemoryState,
    ) -> float:
        if not self.config.use_time_buckets:
            return 1.0
        if bucket_ids is None or state.bucket_memory is None or state.bucket_initialized is None:
            return 0.0

        current_memory, current_available = self._gather_bucket_memory(bucket_ids=bucket_ids, state=state)
        if current_memory is None or current_available is None:
            return 0.0

        neighbor_memory, neighbor_available = self._gather_neighbor_memory(bucket_ids=bucket_ids, state=state)
        num_samples = int(targets.shape[0])
        reference_memory = torch.zeros_like(current_memory)
        reference_count = torch.zeros(
            (num_samples, state.global_memory.shape[0]),
            dtype=state.global_memory.dtype,
            device=self.device,
        )

        if neighbor_memory is not None and neighbor_available is not None:
            reference_memory = reference_memory + torch.where(
                neighbor_available.unsqueeze(-1),
                neighbor_memory,
                torch.zeros_like(neighbor_memory),
            )
            reference_count = reference_count + neighbor_available.to(reference_count.dtype)

        global_available = state.global_initialized.unsqueeze(0).expand(num_samples, -1)
        global_memory = state.global_memory.unsqueeze(0).expand(num_samples, -1, -1)
        reference_memory = reference_memory + torch.where(
            global_available.unsqueeze(-1),
            global_memory,
            torch.zeros_like(global_memory),
        )
        reference_count = reference_count + global_available.to(reference_count.dtype)

        reference_available = reference_count > 0
        if not torch.any(reference_available):
            return 0.0
        reference_memory = reference_memory / reference_count.clamp_min(1.0).unsqueeze(-1)
        reference_memory = self._normalize_memory_rows(
            memory=reference_memory,
            available=reference_available,
        )

        sample_index = torch.arange(targets.shape[0], device=targets.device)
        valid = current_available[sample_index, targets] & reference_available[sample_index, targets]
        if not torch.any(valid):
            return 0.0

        target_index = targets[valid]
        current_vectors = current_memory[valid, target_index]
        reference_vectors = reference_memory[valid, target_index]
        similarity = F.cosine_similarity(current_vectors, reference_vectors, dim=-1)
        similarity = torch.clamp(0.5 * (similarity + 1.0), 0.0, 1.0)
        return float(similarity.mean().detach().item())

    def _build_logits(
        self,
        *,
        normalized_embedding: torch.Tensor,
        bucket_ids: torch.Tensor | None,
        state: PrototypeMemoryState,
    ) -> dict[str, torch.Tensor]:
        global_logits = torch.matmul(normalized_embedding, state.global_memory.t())
        global_available = state.global_initialized.unsqueeze(0).expand_as(global_logits)

        if not self.config.use_time_buckets:
            masked_logits = global_logits.clone()
            masked_logits[~global_available] = -1e4
            return {
                "logits": masked_logits / max(float(self.config.temperature), 1e-3),
                "available": global_available,
            }

        if bucket_ids is None or state.bucket_memory is None or state.bucket_initialized is None:
            raise ValueError("Time-bucket prototype memory requires per-sample bucket ids.")

        bucket_memory, bucket_available = self._gather_bucket_memory(bucket_ids=bucket_ids, state=state)
        if bucket_memory is None or bucket_available is None:
            raise ValueError("Time-bucket prototype memory requires per-sample bucket ids.")
        bucket_logits = torch.einsum("nd,ncd->nc", normalized_embedding, bucket_memory)
        combined_logits = torch.where(bucket_available, bucket_logits, global_logits)
        combined_available = bucket_available | global_available

        neighbor_memory, neighbor_available = self._gather_neighbor_memory(bucket_ids=bucket_ids, state=state)
        neighbor_blend = float(np.clip(self.config.neighbor_blend, 0.0, 1.0))
        if neighbor_memory is not None and neighbor_available is not None:
            neighbor_logits = torch.einsum("nd,ncd->nc", normalized_embedding, neighbor_memory)
            prev_logits = combined_logits
            prev_available = combined_available
            combined_logits = torch.where(neighbor_available & ~prev_available, neighbor_logits, prev_logits)
            if neighbor_blend > 0.0:
                blended_logits = (1.0 - neighbor_blend) * prev_logits + neighbor_blend * neighbor_logits
                combined_logits = torch.where(neighbor_available & prev_available, blended_logits, combined_logits)
            combined_available = prev_available | neighbor_available

        blend = float(np.clip(self.config.global_blend, 0.0, 1.0))
        if blend > 0.0:
            blended_logits = (1.0 - blend) * combined_logits + blend * global_logits
            combined_logits = torch.where(global_available & combined_available, blended_logits, combined_logits)
        combined_logits = combined_logits.clone()
        combined_logits[~combined_available] = -1e4
        return {
            "logits": combined_logits / max(float(self.config.temperature), 1e-3),
            "available": combined_available,
        }


@dataclass(frozen=True)
class TemporalNormalAlignmentConfig:
    embedding_dim: int
    num_buckets: int
    momentum: float = 0.9
    start_epoch: int = 1
    neighbor_blend: float = 0.0
    global_blend: float = 0.0


@dataclass
class TemporalNormalAlignmentState:
    global_memory: torch.Tensor
    global_initialized: torch.Tensor
    bucket_memory: torch.Tensor
    bucket_initialized: torch.Tensor


class TemporalNormalAlignmentBank:
    def __init__(
        self,
        *,
        config: TemporalNormalAlignmentConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        embedding_dim = int(config.embedding_dim)
        num_buckets = max(int(config.num_buckets), 1)
        self.state = TemporalNormalAlignmentState(
            global_memory=torch.zeros((embedding_dim,), dtype=torch.float32, device=device),
            global_initialized=torch.zeros((), dtype=torch.bool, device=device),
            bucket_memory=torch.zeros((num_buckets, embedding_dim), dtype=torch.float32, device=device),
            bucket_initialized=torch.zeros((num_buckets,), dtype=torch.bool, device=device),
        )

    def compute_loss(
        self,
        *,
        embedding: torch.Tensor,
        raw_labels: torch.Tensor,
        bucket_ids: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        normal_mask = raw_labels == 0
        if not torch.any(normal_mask):
            return embedding.new_tensor(0.0)

        normalized_embedding = F.normalize(embedding[normal_mask], dim=-1)
        valid_bucket_ids = bucket_ids[normal_mask].to(dtype=torch.long).clamp_(
            0, self.state.bucket_memory.shape[0] - 1
        )
        loss_state = self._snapshot_state()

        bucket_losses: list[torch.Tensor] = []
        bucket_weights: list[float] = []
        for bucket_value in torch.unique(valid_bucket_ids).tolist():
            bucket_id = int(bucket_value)
            bucket_mask = valid_bucket_ids == bucket_id
            if not torch.any(bucket_mask):
                continue
            bucket_center = normalized_embedding[bucket_mask].mean(dim=0)
            bucket_center = F.normalize(bucket_center.unsqueeze(0), dim=-1).squeeze(0)
            anchor = self._build_anchor(bucket_id=bucket_id, state=loss_state)
            if anchor is not None and int(epoch) >= max(int(self.config.start_epoch), 1):
                bucket_losses.append(1.0 - torch.sum(bucket_center * anchor, dim=-1))
                bucket_weights.append(float(torch.sum(bucket_mask).item()))

        self._update_state(
            normalized_embedding=normalized_embedding,
            bucket_ids=valid_bucket_ids,
        )

        if not bucket_losses:
            return embedding.new_tensor(0.0)
        loss_tensor = torch.stack(bucket_losses, dim=0)
        weight_tensor = torch.as_tensor(
            bucket_weights,
            dtype=loss_tensor.dtype,
            device=loss_tensor.device,
        )
        weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
        return torch.sum(loss_tensor * weight_tensor)

    def build_shift(
        self,
        *,
        bucket_ids: torch.Tensor,
        strength: float,
    ) -> torch.Tensor | None:
        shift_strength = float(np.clip(strength, 0.0, 10.0))
        if shift_strength <= 0.0:
            return None
        state = self._snapshot_state()
        if state.bucket_memory.numel() == 0:
            return None
        clamped_bucket_ids = bucket_ids.to(dtype=torch.long).clamp_(0, state.bucket_memory.shape[0] - 1)
        shift = torch.zeros(
            (int(clamped_bucket_ids.shape[0]), int(state.global_memory.shape[0])),
            dtype=state.global_memory.dtype,
            device=self.device,
        )
        available = torch.zeros((int(clamped_bucket_ids.shape[0]),), dtype=torch.bool, device=self.device)
        for bucket_value in torch.unique(clamped_bucket_ids).tolist():
            bucket_id = int(bucket_value)
            if not bool(state.bucket_initialized[bucket_id].item()):
                continue
            reference = self._build_reference_anchor(bucket_id=bucket_id, state=state)
            if reference is None:
                continue
            delta = reference - state.bucket_memory[bucket_id]
            delta_norm = delta.norm().clamp_min(1e-6)
            delta = delta / delta_norm
            bucket_mask = clamped_bucket_ids == bucket_id
            shift[bucket_mask] = delta.unsqueeze(0)
            available[bucket_mask] = True
        if not torch.any(available):
            return None
        shift[~available] = 0.0
        return shift_strength * shift

    def _snapshot_state(self) -> TemporalNormalAlignmentState:
        return TemporalNormalAlignmentState(
            global_memory=self.state.global_memory.detach().clone(),
            global_initialized=self.state.global_initialized.clone(),
            bucket_memory=self.state.bucket_memory.detach().clone(),
            bucket_initialized=self.state.bucket_initialized.clone(),
        )

    def _build_anchor(
        self,
        *,
        bucket_id: int,
        state: TemporalNormalAlignmentState,
    ) -> torch.Tensor | None:
        components: list[tuple[float, torch.Tensor]] = []
        base_weight = float(
            np.clip(
                1.0 - float(self.config.neighbor_blend) - float(self.config.global_blend),
                0.0,
                1.0,
            )
        )
        if bool(state.bucket_initialized[bucket_id].item()):
            components.append((base_weight if base_weight > 0.0 else 1.0, state.bucket_memory[bucket_id]))

        neighbor_memory = self._neighbor_memory(bucket_id=bucket_id, state=state)
        if neighbor_memory is not None and float(self.config.neighbor_blend) > 0.0:
            components.append((float(self.config.neighbor_blend), neighbor_memory))

        if bool(state.global_initialized.item()) and float(self.config.global_blend) > 0.0:
            components.append((float(self.config.global_blend), state.global_memory))

        if not components:
            if neighbor_memory is not None:
                components.append((1.0, neighbor_memory))
            elif bool(state.global_initialized.item()):
                components.append((1.0, state.global_memory))
            elif bool(state.bucket_initialized[bucket_id].item()):
                components.append((1.0, state.bucket_memory[bucket_id]))

        if not components:
            return None

        total_weight = sum(weight for weight, _ in components)
        anchor = torch.zeros_like(state.global_memory)
        for weight, vector in components:
            anchor = anchor + (weight / max(total_weight, 1e-6)) * vector
        return F.normalize(anchor.unsqueeze(0), dim=-1).squeeze(0)

    def _build_reference_anchor(
        self,
        *,
        bucket_id: int,
        state: TemporalNormalAlignmentState,
    ) -> torch.Tensor | None:
        components: list[tuple[float, torch.Tensor]] = []
        neighbor_memory = self._neighbor_memory(bucket_id=bucket_id, state=state)
        if neighbor_memory is not None:
            weight = float(self.config.neighbor_blend) if float(self.config.neighbor_blend) > 0.0 else 1.0
            components.append((weight, neighbor_memory))
        if bool(state.global_initialized.item()):
            weight = float(self.config.global_blend) if float(self.config.global_blend) > 0.0 else 1.0
            components.append((weight, state.global_memory))
        if not components:
            return None
        total_weight = sum(weight for weight, _ in components)
        anchor = torch.zeros_like(state.global_memory)
        for weight, vector in components:
            anchor = anchor + (weight / max(total_weight, 1e-6)) * vector
        return F.normalize(anchor.unsqueeze(0), dim=-1).squeeze(0)

    def _neighbor_memory(
        self,
        *,
        bucket_id: int,
        state: TemporalNormalAlignmentState,
    ) -> torch.Tensor | None:
        vectors: list[torch.Tensor] = []
        if bucket_id > 0 and bool(state.bucket_initialized[bucket_id - 1].item()):
            vectors.append(state.bucket_memory[bucket_id - 1])
        if bucket_id < (state.bucket_memory.shape[0] - 1) and bool(
            state.bucket_initialized[bucket_id + 1].item()
        ):
            vectors.append(state.bucket_memory[bucket_id + 1])
        if not vectors:
            return None
        neighbor = torch.stack(vectors, dim=0).mean(dim=0)
        return F.normalize(neighbor.unsqueeze(0), dim=-1).squeeze(0)

    def _update_state(
        self,
        *,
        normalized_embedding: torch.Tensor,
        bucket_ids: torch.Tensor,
    ) -> None:
        global_center = normalized_embedding.mean(dim=0)
        global_center = F.normalize(global_center.unsqueeze(0), dim=-1).squeeze(0)
        self._ema_update_global(global_center)

        for bucket_value in torch.unique(bucket_ids).tolist():
            bucket_id = int(bucket_value)
            bucket_mask = bucket_ids == bucket_id
            if not torch.any(bucket_mask):
                continue
            bucket_center = normalized_embedding[bucket_mask].mean(dim=0)
            bucket_center = F.normalize(bucket_center.unsqueeze(0), dim=-1).squeeze(0)
            self._ema_update_bucket(bucket_id=bucket_id, bucket_center=bucket_center)

    def _ema_update_global(self, center: torch.Tensor) -> None:
        momentum = float(np.clip(self.config.momentum, 0.0, 0.999))
        with torch.no_grad():
            if bool(self.state.global_initialized.item()):
                blended = momentum * self.state.global_memory + (1.0 - momentum) * center
                self.state.global_memory = F.normalize(blended.unsqueeze(0), dim=-1).squeeze(0)
            else:
                self.state.global_memory = center
                self.state.global_initialized = torch.ones_like(self.state.global_initialized)

    def _ema_update_bucket(
        self,
        *,
        bucket_id: int,
        bucket_center: torch.Tensor,
    ) -> None:
        momentum = float(np.clip(self.config.momentum, 0.0, 0.999))
        with torch.no_grad():
            if bool(self.state.bucket_initialized[bucket_id].item()):
                blended = momentum * self.state.bucket_memory[bucket_id] + (1.0 - momentum) * bucket_center
                self.state.bucket_memory[bucket_id] = F.normalize(blended.unsqueeze(0), dim=-1).squeeze(0)
            else:
                self.state.bucket_memory[bucket_id] = bucket_center
                self.state.bucket_initialized[bucket_id] = True
