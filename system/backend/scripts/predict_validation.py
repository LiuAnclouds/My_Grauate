from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = EXPERIMENT_ROOT / "src"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DyRIFT-TGAT full-model inference for validation nodes.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--node-ids-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["GRADPROJ_ACTIVE_DATASET"] = str(args.dataset)
    for import_root in (SRC_ROOT, EXPERIMENT_ROOT):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    import torch

    from dyrift.config_loader import resolve_train_parameters
    from dyrift.models.graph import get_experiment_cls
    from dyrift.models.presets import apply_cfg_overrides, build_graph_cfg
    from dyrift.models.runtime import build_runtime
    from dyrift.models.spec import OFFICIAL_TRAIN_EPOCHS
    from dyrift.utils.common import TRAIN_OUTPUT_ROOT, load_experiment_split

    node_payload = json.loads(args.node_ids_file.read_text(encoding="utf-8-sig"))
    node_ids = np.asarray(node_payload["node_ids"], dtype=np.int32)
    parameter_file = EXPERIMENT_ROOT / "configs" / "train" / f"{args.dataset}.json"
    params = resolve_train_parameters(
        args=SimpleNamespace(parameter_file=parameter_file),
        default_epochs=OFFICIAL_TRAIN_EPOCHS,
        default_outdir=TRAIN_OUTPUT_ROOT,
    )
    split = load_experiment_split()
    graph_config = build_graph_cfg(params.model, params.preset)
    graph_config, applied_overrides = apply_cfg_overrides(
        graph_config,
        params.graph_config_overrides,
    )
    runtime_updates = {}
    if params.learning_rate is not None:
        runtime_updates["learning_rate"] = float(params.learning_rate)
    if params.weight_decay is not None:
        runtime_updates["weight_decay"] = float(params.weight_decay)
    if params.dropout is not None:
        runtime_updates["dropout"] = float(params.dropout)
    if runtime_updates:
        graph_config = replace(graph_config, **runtime_updates)
        applied_overrides = {**applied_overrides, **runtime_updates}

    runtime = build_runtime(
        feature_dir=params.feature_dir,
        model_name=params.model,
        split=split,
        train_ids=split.train_ids,
        graph_config=graph_config,
        feature_profile=params.feature_profile,
        target_context_groups=params.target_context_groups,
    )
    experiment_cls = get_experiment_cls(params.model)
    experiment = experiment_cls(
        model_name=params.model,
        seed=int(params.seeds[0]),
        input_dim=runtime.input_dim,
        num_relations=runtime.num_relations,
        max_day=runtime.global_max_day,
        feature_groups=runtime.feature_groups,
        hidden_dim=params.hidden_dim,
        num_layers=len(params.fanouts),
        rel_dim=params.rel_dim,
        fanouts=list(params.fanouts),
        batch_size=params.batch_size,
        epochs=params.epochs,
        device=args.device,
        graph_config=runtime.graph_config,
        feature_normalizer_state=runtime.feature_normalizer_state,
        target_context_input_dim=runtime.target_context_input_dim,
        target_context_feature_groups=runtime.target_context_feature_groups,
        target_context_normalizer_state=runtime.target_context_normalizer_state,
    )
    run_dir = params.outdir / params.experiment_name / params.model / str(args.dataset)
    checkpoint_path = run_dir / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "model.pt"
    state_dict = torch.load(checkpoint_path, map_location=experiment.device, weights_only=True)
    missing, unexpected = experiment.network.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )
    scores = experiment.predict_proba(
        runtime.phase1_context,
        node_ids,
        batch_seed=int(params.seeds[0]) + 1000,
        show_progress=False,
    )
    payload = {
        "dataset": str(args.dataset),
        "model": "DyRIFT-TGAT",
        "checkpoint": str(checkpoint_path),
        "node_count": int(node_ids.size),
        "results": [
            {"node_id": str(int(node_id)), "risk_score": float(score)}
            for node_id, score in zip(node_ids.tolist(), scores.tolist(), strict=True)
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
