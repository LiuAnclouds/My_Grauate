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

    def __post_init__(self) -> None:
        if self.bucket_mode not in {"global", "time_bucket"}:
            raise ValueError(f"Unsupported prototype bucket mode: {self.bucket_mode}")

    @property
    def enabled(self) -> bool:
        return int(self.num_classes) >= 3

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

    def active(self, epoch: int) -> bool:
        return self.config.enabled and int(epoch) >= max(int(self.config.start_epoch), 1)

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
        if not self.active(epoch):
            return embedding.new_tensor(0.0)

        valid_mask = targets >= 0
        if not torch.any(valid_mask):
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
                total_loss = total_loss + consistency_weight_value * consistency_loss.mean()

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
