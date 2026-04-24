from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dyrift.data_processing.core.registry import get_active_dataset_spec
from dyrift.analysis.data_loader import LABEL_NAMES, PhaseData, load_phase


ALL_ANALYSES = ("overview", "feature", "graph", "temporal", "drift", "split")
PLOT_SAMPLE_SIZE = 50_000
DRIFT_BIN_COUNT = 10
FEATURE_QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
TIME_WINDOW_COUNT = 4
ACTIVE_DATASET_SPEC = get_active_dataset_spec()


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_square_csv(path: Path, headers: list[str], matrix: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["feature"] + headers)
        for idx, header in enumerate(headers):
            writer.writerow([header] + [float(value) for value in matrix[idx].tolist()])


def label_name(label: int) -> str:
    return LABEL_NAMES.get(int(label), f"label_{int(label)}")


def background_labels() -> tuple[int, ...]:
    return tuple(int(label) for label in ACTIVE_DATASET_SPEC.background_labels)


def split_train_artifact() -> str:
    return str(ACTIVE_DATASET_SPEC.split_train_artifact)


def split_val_artifact() -> str:
    return str(ACTIVE_DATASET_SPEC.split_val_artifact)


def split_external_artifact() -> str | None:
    return ACTIVE_DATASET_SPEC.split_external_artifact


def background_group_specs(labels: np.ndarray) -> list[tuple[str, np.ndarray]]:
    specs: list[tuple[str, np.ndarray]] = []
    for label in background_labels():
        specs.append(
            (
                label_name(label),
                np.flatnonzero(labels == label).astype(np.int32, copy=False),
            )
        )
    return specs


def label_order_for_phase(labels: np.ndarray) -> list[int]:
    ordered: list[int] = []
    for label in [-100, 0, 1, *background_labels()]:
        if np.any(labels == label):
            ordered.append(int(label))
    for label in sorted(np.unique(labels).tolist()):
        label = int(label)
        if label not in ordered:
            ordered.append(label)
    return ordered


def quantile_row(prefix: str, values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, FEATURE_QUANTILES)
    names = ("min", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "max")
    return {
        f"{prefix}_{name}": float(value)
        for name, value in zip(names, quantiles.tolist(), strict=True)
    }


def basic_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def sample_values(values: np.ndarray, seed: int, max_size: int = PLOT_SAMPLE_SIZE) -> np.ndarray:
    if values.size <= max_size:
        return values
    rng = np.random.default_rng(seed)
    indices = rng.choice(values.size, size=max_size, replace=False)
    return values[indices]


def plot_empirical_cdf(ax: plt.Axes, values: np.ndarray, label: str, color: str) -> None:
    if values.size == 0:
        return
    sample = sample_values(values, seed=values.size)
    sorted_values = np.sort(sample)
    y = np.linspace(1.0 / sorted_values.size, 1.0, num=sorted_values.size)
    ax.plot(sorted_values, y, label=label, color=color, linewidth=1.8)


def build_time_windows(values: np.ndarray, n_windows: int = TIME_WINDOW_COUNT) -> list[dict[str, Any]]:
    raw_edges = np.quantile(values, np.linspace(0.0, 1.0, num=n_windows + 1))
    edges = raw_edges.astype(np.float64, copy=True)
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    windows: list[dict[str, Any]] = []
    for idx in range(n_windows):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == n_windows - 1:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
        windows.append(
            {
                "window_idx": idx + 1,
                "label": f"W{idx + 1}",
                "start_day": int(np.floor(raw_edges[idx])),
                "end_day": int(np.ceil(raw_edges[idx + 1])),
                "mask": mask,
            }
        )
    return windows


def build_phase_output_dir(outdir: Path, phase: str) -> Path:
    return ensure_dir(outdir / phase)


def get_train_target(data: PhaseData) -> tuple[np.ndarray, np.ndarray]:
    train_labels = data.y[data.train_mask]
    return data.train_mask, train_labels


def analyze_overview(data: PhaseData, outdir: Path) -> dict[str, Any]:
    phase_dir = build_phase_output_dir(outdir, data.phase)
    label_values, label_counts = np.unique(data.y, return_counts=True)
    label_count_map = {
        label_name(label): int(count)
        for label, count in zip(label_values.tolist(), label_counts.tolist(), strict=True)
    }
    edge_type_counts = np.bincount(data.edge_type, minlength=int(data.edge_type.max()) + 1)[1:]
    edge_type_map = {
        f"type_{idx}": int(count)
        for idx, count in enumerate(edge_type_counts.tolist(), start=1)
    }
    train_labels = data.y[data.train_mask]
    pos_count = int(np.sum(train_labels == 1))
    neg_count = int(np.sum(train_labels == 0))

    summary = {
        "dataset_path": str(data.path),
        "num_nodes": data.num_nodes,
        "num_edges": data.num_edges,
        "num_features": int(data.x.shape[1]),
        "label_counts": label_count_map,
        "train_size": int(data.train_mask.size),
        "test_size": int(data.test_mask.size),
        "train_positive_count": pos_count,
        "train_negative_count": neg_count,
        "train_positive_rate": float(pos_count / max(train_labels.size, 1)),
        "train_neg_pos_ratio": float(neg_count / max(pos_count, 1)),
        "edge_type_counts": edge_type_map,
        "edge_timestamp_min": int(data.edge_timestamp.min()),
        "edge_timestamp_max": int(data.edge_timestamp.max()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(list(label_count_map.keys()), list(label_count_map.values()), color="#2c7fb8")
    axes[0].set_title(f"{data.phase} 节点标签分布")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("节点数")

    axes[1].bar(list(edge_type_map.keys()), list(edge_type_map.values()), color="#41ab5d")
    axes[1].set_title(f"{data.phase} 边类型频次")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_ylabel("边数")

    fig.tight_layout()
    fig.savefig(phase_dir / "overview_distribution.png", dpi=180)
    plt.close(fig)
    return summary


def analyze_features(
    data: PhaseData,
    outdir: Path,
    temporal_core: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_dir = build_phase_output_dir(outdir, data.phase)
    if temporal_core is None:
        temporal_core = compute_temporal_core(data)
    rows: list[dict[str, Any]] = []
    train_idx, train_labels = get_train_target(data)
    train_x = data.x[train_idx]
    normal_mask = train_labels == 0
    fraud_mask = train_labels == 1
    neg1_ratios: list[float] = []
    normal_means: list[float] = []
    fraud_means: list[float] = []
    mean_gaps: list[float] = []

    for feature_idx in range(data.x.shape[1]):
        col = data.x[:, feature_idx]
        train_col = train_x[:, feature_idx]
        normal_col = train_col[normal_mask]
        fraud_col = train_col[fraud_mask]
        neg1_ratio_all = float(np.mean(col == -1))
        normal_mean = float(np.mean(normal_col))
        fraud_mean = float(np.mean(fraud_col))
        mean_gap = abs(normal_mean - fraud_mean)
        row = {
            "phase": data.phase,
            "feature_idx": feature_idx,
            "feature_name": f"x{feature_idx}",
            "count_all": int(col.size),
            "count_train_target": int(train_col.size),
            "neg1_ratio_all": neg1_ratio_all,
            "mean_all": float(np.mean(col)),
            "std_all": float(np.std(col)),
            "train_normal_mean": normal_mean,
            "train_fraud_mean": fraud_mean,
            "train_mean_gap_abs": float(mean_gap),
            "train_normal_neg1_ratio": float(np.mean(normal_col == -1)),
            "train_fraud_neg1_ratio": float(np.mean(fraud_col == -1)),
        }
        row.update(quantile_row("all", col))
        row.update(quantile_row("train_target", train_col))
        rows.append(row)
        neg1_ratios.append(neg1_ratio_all)
        normal_means.append(normal_mean)
        fraud_means.append(fraud_mean)
        mean_gaps.append(float(mean_gap))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([f"x{i}" for i in range(len(neg1_ratios))], neg1_ratios, color="#ef6548")
    ax.set_title(f"{data.phase} 特征中 -1 哨兵值占比")
    ax.set_xlabel("特征")
    ax.set_ylabel("占比")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(phase_dir / "feature_neg1_ratio.png", dpi=180)
    plt.close(fig)

    top_features = np.argsort(-np.asarray(mean_gaps))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, feature_idx in zip(axes.flat, top_features.tolist(), strict=False):
        normal_sample = sample_values(train_x[normal_mask, feature_idx], seed=feature_idx + 1)
        fraud_sample = sample_values(train_x[fraud_mask, feature_idx], seed=feature_idx + 101)
        axis.hist(normal_sample, bins=50, alpha=0.65, label="normal", density=True)
        axis.hist(fraud_sample, bins=50, alpha=0.65, label="fraud", density=True)
        axis.set_title(f"{data.phase} x{feature_idx} 训练分布")
        axis.legend()
    fig.tight_layout()
    fig.savefig(phase_dir / "feature_top_gap_hist.png", dpi=180)
    plt.close(fig)

    missing_group_specs = [
        ("train_normal", data.train_mask[data.y[data.train_mask] == 0]),
        ("train_fraud", data.train_mask[data.y[data.train_mask] == 1]),
        ("test_holdout", data.test_mask),
    ]
    missing_group_specs[2:2] = background_group_specs(data.y)
    missing_rows: list[dict[str, Any]] = []
    missing_matrix: list[list[float]] = []
    missing_labels: list[str] = []
    for group_name, group_idx in missing_group_specs:
        group_ratios: list[float] = []
        for feature_idx in range(data.x.shape[1]):
            group_values = data.x[group_idx, feature_idx]
            neg1_ratio = float(np.mean(group_values == -1))
            missing_rows.append(
                {
                    "phase": data.phase,
                    "group": group_name,
                    "feature_idx": feature_idx,
                    "feature_name": f"x{feature_idx}",
                    "neg1_ratio": neg1_ratio,
                    "mean": float(np.mean(group_values)),
                    "median": float(np.median(group_values)),
                }
            )
            group_ratios.append(neg1_ratio)
        missing_matrix.append(group_ratios)
        missing_labels.append(group_name)
    write_csv(phase_dir / "feature_missing_by_group.csv", missing_rows)

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(np.asarray(missing_matrix), aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title(f"{data.phase} feature missing ratio by group")
    ax.set_xlabel("feature")
    ax.set_ylabel("group")
    ax.set_xticks(np.arange(data.x.shape[1]))
    ax.set_xticklabels([f"x{i}" for i in range(data.x.shape[1])], rotation=30)
    ax.set_yticks(np.arange(len(missing_labels)))
    ax.set_yticklabels(missing_labels)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(phase_dir / "feature_missing_by_label.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cdf_colors = {"normal": "#2c7fb8", "fraud": "#ef6548"}
    for axis, feature_idx in zip(axes.flat, top_features.tolist(), strict=False):
        plot_empirical_cdf(
            axis,
            train_x[normal_mask, feature_idx],
            label="normal",
            color=cdf_colors["normal"],
        )
        plot_empirical_cdf(
            axis,
            train_x[fraud_mask, feature_idx],
            label="fraud",
            color=cdf_colors["fraud"],
        )
        axis.set_title(f"{data.phase} x{feature_idx} empirical CDF")
        axis.set_xlabel("feature value")
        axis.set_ylabel("cumulative probability")
        axis.legend()
    fig.tight_layout()
    fig.savefig(phase_dir / "feature_top_gap_cdf.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, feature_idx in zip(axes.flat, top_features.tolist(), strict=False):
        normal_sample = sample_values(train_x[normal_mask, feature_idx], seed=feature_idx + 7)
        fraud_sample = sample_values(train_x[fraud_mask, feature_idx], seed=feature_idx + 107)
        axis.boxplot(
            [normal_sample, fraud_sample],
            labels=["normal", "fraud"],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#9ecae1"},
            medianprops={"color": "#cb181d"},
        )
        axis.set_title(f"{data.phase} x{feature_idx} boxplot")
        axis.set_ylabel("feature value")
    fig.tight_layout()
    fig.savefig(phase_dir / "feature_boxplots_topk.png", dpi=180)
    plt.close(fig)

    feature_headers = [f"x{i}" for i in range(data.x.shape[1])]
    feature_corr = np.corrcoef(train_x, rowvar=False)
    write_square_csv(phase_dir / "feature_corr_train.csv", feature_headers, feature_corr)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(feature_corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(f"{data.phase} train feature correlation")
    ax.set_xticks(np.arange(data.x.shape[1]))
    ax.set_yticks(np.arange(data.x.shape[1]))
    ax.set_xticklabels(feature_headers, rotation=30)
    ax.set_yticklabels(feature_headers)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(phase_dir / "feature_corr_train.png", dpi=180)
    plt.close(fig)

    first_active_train = temporal_core["first_active"][train_idx]
    time_windows = build_time_windows(first_active_train)
    window_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, len(top_features), figsize=(4.2 * len(top_features), 8), squeeze=False)
    for col_idx, feature_idx in enumerate(top_features.tolist()):
        normal_medians: list[float] = []
        fraud_medians: list[float] = []
        normal_missing: list[float] = []
        fraud_missing: list[float] = []
        x_labels: list[str] = []
        for window in time_windows:
            mask_window = window["mask"]
            window_values = train_x[mask_window, feature_idx]
            window_labels = train_labels[mask_window]
            for group_name, group_mask in (("normal", window_labels == 0), ("fraud", window_labels == 1)):
                group_values = window_values[group_mask]
                if group_values.size == 0:
                    median_value = float("nan")
                    neg1_ratio = float("nan")
                else:
                    median_value = float(np.median(group_values))
                    neg1_ratio = float(np.mean(group_values == -1))
                window_rows.append(
                    {
                        "phase": data.phase,
                        "feature_idx": feature_idx,
                        "feature_name": f"x{feature_idx}",
                        "window_idx": window["window_idx"],
                        "window_label": window["label"],
                        "window_start_day": window["start_day"],
                        "window_end_day": window["end_day"],
                        "group": group_name,
                        "count": int(group_values.size),
                        "median": median_value,
                        "mean": float(np.mean(group_values)) if group_values.size else float("nan"),
                        "neg1_ratio": neg1_ratio,
                    }
                )
                if group_name == "normal":
                    normal_medians.append(median_value)
                    normal_missing.append(neg1_ratio)
                else:
                    fraud_medians.append(median_value)
                    fraud_missing.append(neg1_ratio)
            x_labels.append(window["label"])

        axes[0, col_idx].plot(x_labels, normal_medians, marker="o", label="normal")
        axes[0, col_idx].plot(x_labels, fraud_medians, marker="o", label="fraud")
        axes[0, col_idx].set_title(f"{data.phase} x{feature_idx} median by time window")
        axes[0, col_idx].set_xlabel("time window")
        axes[0, col_idx].set_ylabel("median")
        axes[0, col_idx].legend()

        axes[1, col_idx].plot(x_labels, normal_missing, marker="o", label="normal")
        axes[1, col_idx].plot(x_labels, fraud_missing, marker="o", label="fraud")
        axes[1, col_idx].set_title(f"{data.phase} x{feature_idx} missing ratio by time window")
        axes[1, col_idx].set_xlabel("time window")
        axes[1, col_idx].set_ylabel("neg1 ratio")
        axes[1, col_idx].set_ylim(0.0, 1.0)
        axes[1, col_idx].legend()
    fig.tight_layout()
    fig.savefig(phase_dir / "feature_time_drift_by_window.png", dpi=180)
    plt.close(fig)
    write_csv(phase_dir / "feature_time_window_profile.csv", window_rows)

    compact = {
        "neg1_ratio_all": neg1_ratios,
        "train_normal_mean": normal_means,
        "train_fraud_mean": fraud_means,
        "train_mean_gap_abs": mean_gaps,
        "top_gap_features": [int(idx) for idx in top_features.tolist()],
        "additional_outputs": [
            "feature_missing_by_label.png",
            "feature_top_gap_cdf.png",
            "feature_boxplots_topk.png",
            "feature_corr_train.png",
            "feature_time_drift_by_window.png",
            "feature_missing_by_group.csv",
            "feature_corr_train.csv",
            "feature_time_window_profile.csv",
        ],
    }
    return rows, compact


def compute_degree_arrays(data: PhaseData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = data.edge_index[:, 0]
    dst = data.edge_index[:, 1]
    indegree = np.bincount(dst, minlength=data.num_nodes).astype(np.int32, copy=False)
    outdegree = np.bincount(src, minlength=data.num_nodes).astype(np.int32, copy=False)
    total_degree = indegree + outdegree
    return indegree, outdegree, total_degree


def compute_temporal_core(data: PhaseData) -> dict[str, np.ndarray]:
    max_day = int(data.edge_timestamp.max())
    edge_count_by_day = np.bincount(data.edge_timestamp, minlength=max_day + 1)[1:]

    edge_type_day = np.zeros((int(data.edge_type.max()), max_day), dtype=np.int32)
    np.add.at(edge_type_day, (data.edge_type - 1, data.edge_timestamp - 1), 1)

    src = data.edge_index[:, 0]
    dst = data.edge_index[:, 1]
    first_active = np.full(data.num_nodes, max_day + 1, dtype=np.int32)
    last_active = np.zeros(data.num_nodes, dtype=np.int32)
    np.minimum.at(first_active, src, data.edge_timestamp)
    np.minimum.at(first_active, dst, data.edge_timestamp)
    np.maximum.at(last_active, src, data.edge_timestamp)
    np.maximum.at(last_active, dst, data.edge_timestamp)
    active_span = last_active - first_active

    return {
        "edge_count_by_day": edge_count_by_day,
        "edge_type_day": edge_type_day,
        "first_active": first_active,
        "last_active": last_active,
        "active_span": active_span,
    }


def analyze_graph(
    data: PhaseData,
    outdir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_dir = build_phase_output_dir(outdir, data.phase)
    indegree, outdegree, total_degree = compute_degree_arrays(data)
    rows: list[dict[str, Any]] = []
    labels = data.y
    groups = {
        "all_nodes": np.arange(data.num_nodes, dtype=np.int32),
        "train_normal": data.train_mask[labels[data.train_mask] == 0],
        "train_fraud": data.train_mask[labels[data.train_mask] == 1],
        "test_holdout": data.test_mask,
    }
    for group_name, group_idx in background_group_specs(labels):
        groups[group_name] = group_idx

    for group_name, group_idx in groups.items():
        if group_idx.size == 0:
            continue
        degree_stats = {
            "indegree": indegree[group_idx],
            "outdegree": outdegree[group_idx],
            "total_degree": total_degree[group_idx],
        }
        for degree_name, values in degree_stats.items():
            stats = basic_stats(values)
            for metric_name, metric_value in stats.items():
                rows.append(
                    {
                        "phase": data.phase,
                        "entity": "degree",
                        "group": group_name,
                        "metric": f"{degree_name}_{metric_name}",
                        "value": metric_value,
                    }
                )

    edge_type_counts = np.bincount(data.edge_type, minlength=int(data.edge_type.max()) + 1)[1:]
    for edge_type_idx, count in enumerate(edge_type_counts.tolist(), start=1):
        rows.append(
            {
                "phase": data.phase,
                "entity": "edge_type",
                "group": f"type_{edge_type_idx}",
                "metric": "count",
                "value": int(count),
            }
        )
        rows.append(
            {
                "phase": data.phase,
                "entity": "edge_type",
                "group": f"type_{edge_type_idx}",
                "metric": "ratio",
                "value": float(count / max(data.num_edges, 1)),
            }
        )

    label_order = label_order_for_phase(labels)
    label_to_code = {label: idx for idx, label in enumerate(label_order)}
    code_to_label = {idx: label for label, idx in label_to_code.items()}
    node_codes = np.zeros(data.num_nodes, dtype=np.int8)
    for label, code in label_to_code.items():
        node_codes[labels == label] = code
    src_codes = node_codes[data.edge_index[:, 0]]
    dst_codes = node_codes[data.edge_index[:, 1]]
    pair_codes = src_codes.astype(np.int16) * len(label_order) + dst_codes.astype(np.int16)
    pair_counts = np.bincount(pair_codes, minlength=len(label_order) ** 2)
    pair_matrix = pair_counts.reshape(len(label_order), len(label_order))
    for src_code, src_label in code_to_label.items():
        for dst_code, dst_label in code_to_label.items():
            count = int(pair_matrix[src_code, dst_code])
            rows.append(
                {
                    "phase": data.phase,
                    "entity": "edge_label_pair",
                    "group": f"{label_name(src_label)}->{label_name(dst_label)}",
                    "metric": "count",
                    "value": count,
                }
            )
            rows.append(
                {
                    "phase": data.phase,
                    "entity": "edge_label_pair",
                    "group": f"{label_name(src_label)}->{label_name(dst_label)}",
                    "metric": "ratio",
                    "value": float(count / max(data.num_edges, 1)),
                }
            )

    background_mask = np.isin(labels, background_labels())
    train_target_mask = np.isin(labels, (0, 1))
    src = data.edge_index[:, 0]
    dst = data.edge_index[:, 1]
    background_incident_edge_mask = background_mask[src] | background_mask[dst]
    rows.append(
        {
            "phase": data.phase,
            "entity": "background_bridge",
            "group": "background_incident_edges",
            "metric": "count",
            "value": int(np.sum(background_incident_edge_mask)),
        }
    )
    rows.append(
        {
            "phase": data.phase,
            "entity": "background_bridge",
            "group": "background_incident_edges",
            "metric": "ratio",
            "value": float(np.mean(background_incident_edge_mask)),
        }
    )

    touch_background = np.zeros(data.num_nodes, dtype=bool)
    target_from_src = background_mask[src] & train_target_mask[dst]
    target_from_dst = background_mask[dst] & train_target_mask[src]
    touch_background[dst[target_from_src]] = True
    touch_background[src[target_from_dst]] = True
    train_target_nodes = data.train_mask
    rows.append(
        {
            "phase": data.phase,
            "entity": "background_bridge",
            "group": "train_target_nodes_with_background_neighbor",
            "metric": "count",
            "value": int(np.sum(touch_background[train_target_nodes])),
        }
    )
    rows.append(
        {
            "phase": data.phase,
            "entity": "background_bridge",
            "group": "train_target_nodes_with_background_neighbor",
            "metric": "ratio",
            "value": float(np.mean(touch_background[train_target_nodes])),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    normal_degree_sample = sample_values(total_degree[groups["train_normal"]], seed=17)
    fraud_degree_sample = sample_values(total_degree[groups["train_fraud"]], seed=29)
    bins = np.arange(0, int(max(normal_degree_sample.max(), fraud_degree_sample.max())) + 2) - 0.5
    axes[0].hist(normal_degree_sample, bins=bins, alpha=0.7, density=True, label="normal")
    axes[0].hist(fraud_degree_sample, bins=bins, alpha=0.7, density=True, label="fraud")
    axes[0].set_yscale("log")
    axes[0].set_title(f"{data.phase} 训练节点总度分布")
    axes[0].set_xlabel("总度")
    axes[0].set_ylabel("密度(log)")
    axes[0].legend()

    im = axes[1].imshow(pair_matrix, cmap="YlOrRd")
    axes[1].set_title(f"{data.phase} 端点标签对热力图")
    axes[1].set_xticks(np.arange(len(label_order)))
    axes[1].set_yticks(np.arange(len(label_order)))
    axes[1].set_xticklabels([label_name(label) for label in label_order], rotation=30)
    axes[1].set_yticklabels([label_name(label) for label in label_order])
    fig.colorbar(im, ax=axes[1], shrink=0.85)

    fig.tight_layout()
    fig.savefig(phase_dir / "graph_structure.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    direction_groups = {
        "normal": groups["train_normal"],
        "fraud": groups["train_fraud"],
    }
    indegree_means = [float(np.mean(indegree[idx])) for idx in direction_groups.values()]
    outdegree_means = [float(np.mean(outdegree[idx])) for idx in direction_groups.values()]
    indegree_medians = [float(np.median(indegree[idx])) for idx in direction_groups.values()]
    outdegree_medians = [float(np.median(outdegree[idx])) for idx in direction_groups.values()]
    x = np.arange(len(direction_groups))
    width = 0.35

    axes[0, 0].bar(x - width / 2, indegree_means, width=width, label="indegree")
    axes[0, 0].bar(x + width / 2, outdegree_means, width=width, label="outdegree")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(list(direction_groups.keys()))
    axes[0, 0].set_title(f"{data.phase} mean indegree vs outdegree")
    axes[0, 0].set_ylabel("mean degree")
    axes[0, 0].legend()

    axes[0, 1].bar(x - width / 2, indegree_medians, width=width, label="indegree")
    axes[0, 1].bar(x + width / 2, outdegree_medians, width=width, label="outdegree")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(list(direction_groups.keys()))
    axes[0, 1].set_title(f"{data.phase} median indegree vs outdegree")
    axes[0, 1].set_ylabel("median degree")
    axes[0, 1].legend()

    for axis, degree_values, title in (
        (axes[1, 0], indegree, "indegree CDF"),
        (axes[1, 1], outdegree, "outdegree CDF"),
    ):
        for group_name, group_idx in direction_groups.items():
            plot_empirical_cdf(
                axis,
                degree_values[group_idx],
                label=group_name,
                color="#2c7fb8" if group_name == "normal" else "#ef6548",
            )
        axis.set_title(f"{data.phase} {title}")
        axis.set_xlabel("degree")
        axis.set_ylabel("cumulative probability")
        axis.legend()

    fig.tight_layout()
    fig.savefig(phase_dir / "degree_direction.png", dpi=180)
    plt.close(fig)

    compact = {
        "total_degree_mean": float(np.mean(total_degree)),
        "total_degree_p95": float(np.quantile(total_degree, 0.95)),
        "train_normal_degree_mean": float(np.mean(total_degree[groups["train_normal"]])),
        "train_fraud_degree_mean": float(np.mean(total_degree[groups["train_fraud"]])),
        "train_normal_indegree_mean": float(np.mean(indegree[groups["train_normal"]])),
        "train_fraud_indegree_mean": float(np.mean(indegree[groups["train_fraud"]])),
        "train_normal_outdegree_mean": float(np.mean(outdegree[groups["train_normal"]])),
        "train_fraud_outdegree_mean": float(np.mean(outdegree[groups["train_fraud"]])),
        "background_incident_edge_ratio": float(np.mean(background_incident_edge_mask)),
        "train_target_touch_background_ratio": float(np.mean(touch_background[train_target_nodes])),
        "additional_outputs": [
            "graph_structure.png",
            "degree_direction.png",
        ],
    }
    return rows, compact


def analyze_temporal(
    data: PhaseData,
    outdir: Path,
    temporal_core: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    phase_dir = build_phase_output_dir(outdir, data.phase)
    if temporal_core is None:
        temporal_core = compute_temporal_core(data)

    rows: list[dict[str, Any]] = []
    edge_count_by_day = temporal_core["edge_count_by_day"]
    edge_type_day = temporal_core["edge_type_day"]
    first_active = temporal_core["first_active"]
    last_active = temporal_core["last_active"]
    active_span = temporal_core["active_span"]

    for day, count in enumerate(edge_count_by_day.tolist(), start=1):
        rows.append(
            {
                "phase": data.phase,
                "entity": "edge_day",
                "group": f"day_{day}",
                "metric": "count",
                "value": int(count),
            }
        )

    for edge_type_idx in range(edge_type_day.shape[0]):
        for day in range(edge_type_day.shape[1]):
            rows.append(
                {
                    "phase": data.phase,
                    "entity": "edge_type_day",
                    "group": f"type_{edge_type_idx + 1}_day_{day + 1}",
                    "metric": "count",
                    "value": int(edge_type_day[edge_type_idx, day]),
                }
            )

    group_indices = {
        "all_nodes": np.arange(data.num_nodes, dtype=np.int32),
        "train_normal": data.train_mask[data.y[data.train_mask] == 0],
        "train_fraud": data.train_mask[data.y[data.train_mask] == 1],
        "test_holdout": data.test_mask,
    }
    background_nodes = np.flatnonzero(np.isin(data.y, background_labels())).astype(np.int32, copy=False)
    if background_nodes.size:
        group_indices["background_nodes"] = background_nodes
    for group_name, group_idx in group_indices.items():
        for metric_name, values in {
            "first_active": first_active[group_idx],
            "last_active": last_active[group_idx],
            "active_span": active_span[group_idx],
        }.items():
            stats = basic_stats(values)
            for stat_name, stat_value in stats.items():
                rows.append(
                    {
                        "phase": data.phase,
                        "entity": "node_activity",
                        "group": group_name,
                        "metric": f"{metric_name}_{stat_name}",
                        "value": stat_value,
                    }
                )

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(np.arange(1, edge_count_by_day.size + 1), edge_count_by_day, color="#3182bd")
    axes[0].set_title(f"{data.phase} 按天边数量变化")
    axes[0].set_xlabel("相对天数")
    axes[0].set_ylabel("边数")

    im = axes[1].imshow(edge_type_day, aspect="auto", cmap="Blues")
    axes[1].set_title(f"{data.phase} 边类型-时间热力图")
    axes[1].set_ylabel("边类型")
    axes[1].set_xlabel("相对天数")
    axes[1].set_yticks(np.arange(edge_type_day.shape[0]))
    axes[1].set_yticklabels([f"type_{idx}" for idx in range(1, edge_type_day.shape[0] + 1)])
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    normal_first = sample_values(first_active[group_indices["train_normal"]], seed=43)
    fraud_first = sample_values(first_active[group_indices["train_fraud"]], seed=59)
    bins = np.arange(1, int(max(normal_first.max(), fraud_first.max())) + 2) - 0.5
    axes[2].hist(normal_first, bins=bins, alpha=0.7, density=True, label="normal")
    axes[2].hist(fraud_first, bins=bins, alpha=0.7, density=True, label="fraud")
    axes[2].set_title(f"{data.phase} 正常/欺诈节点首次活跃时间")
    axes[2].set_xlabel("首次活跃相对天数")
    axes[2].set_ylabel("密度")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(phase_dir / "temporal_patterns.png", dpi=180)
    plt.close(fig)

    compact = {
        "edge_day_mean": float(np.mean(edge_count_by_day)),
        "edge_day_p95": float(np.quantile(edge_count_by_day, 0.95)),
        "active_days": int(edge_count_by_day.size),
        "train_normal_first_active_median": float(np.median(first_active[group_indices["train_normal"]])),
        "train_fraud_first_active_median": float(np.median(first_active[group_indices["train_fraud"]])),
        "train_normal_active_span_median": float(np.median(active_span[group_indices["train_normal"]])),
        "train_fraud_active_span_median": float(np.median(active_span[group_indices["train_fraud"]])),
    }
    return rows, compact, temporal_core


def _psi(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    ref_ratio = ref_hist / max(ref_hist.sum(), 1)
    cur_ratio = cur_hist / max(cur_hist.sum(), 1)
    eps = 1e-6
    ref_ratio = np.clip(ref_ratio, eps, None)
    cur_ratio = np.clip(cur_ratio, eps, None)
    return float(np.sum((ref_ratio - cur_ratio) * np.log(ref_ratio / cur_ratio)))


def _build_drift_bins(values: np.ndarray, n_bins: int = DRIFT_BIN_COUNT) -> np.ndarray:
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, num=n_bins + 1))
    unique = np.unique(quantiles)
    if unique.size <= 2:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if lower == upper:
            upper = lower + 1e-6
        return np.array([lower - 1e-6, upper + 1e-6], dtype=np.float64)
    unique[0] -= 1e-6
    unique[-1] += 1e-6
    return unique


def analyze_drift(outdir: Path) -> tuple[list[dict[str, Any]], str]:
    if ACTIVE_DATASET_SPEC.split_style != "two_phase":
        summary_text = "# 漂移摘要\n\n- 当前数据集按单图统一协议运行，默认不再生成 phase1 -> phase2 漂移报告。"
        (outdir / "drift_report.md").write_text(summary_text, encoding="utf-8")
        return [], summary_text

    phase1 = load_phase("phase1")
    phase2 = load_phase("phase2")
    phase1_train = phase1.x[phase1.train_mask]
    phase2_train = phase2.x[phase2.train_mask]

    drift_rows: list[dict[str, Any]] = []
    for feature_idx in range(phase1.x.shape[1]):
        ref = phase1_train[:, feature_idx]
        cur = phase2_train[:, feature_idx]
        bins = _build_drift_bins(ref)
        psi = _psi(ref, cur, bins)
        row = {
            "feature_idx": feature_idx,
            "feature_name": f"x{feature_idx}",
            "psi": psi,
            "phase1_train_mean": float(np.mean(ref)),
            "phase2_train_mean": float(np.mean(cur)),
            "phase1_train_median": float(np.median(ref)),
            "phase2_train_median": float(np.median(cur)),
            "phase1_neg1_ratio": float(np.mean(ref == -1)),
            "phase2_neg1_ratio": float(np.mean(cur == -1)),
            "mean_shift": float(np.mean(cur) - np.mean(ref)),
            "median_shift": float(np.median(cur) - np.median(ref)),
        }
        drift_rows.append(row)

    drift_rows.sort(key=lambda row: row["psi"], reverse=True)
    write_csv(outdir / "feature_drift.csv", drift_rows)

    _, _, total_degree_1 = compute_degree_arrays(phase1)
    _, _, total_degree_2 = compute_degree_arrays(phase2)
    temporal_1 = compute_temporal_core(phase1)
    temporal_2 = compute_temporal_core(phase2)
    edge_type_1 = np.bincount(phase1.edge_type, minlength=int(phase1.edge_type.max()) + 1)[1:]
    edge_type_2 = np.bincount(phase2.edge_type, minlength=int(phase2.edge_type.max()) + 1)[1:]
    edge_type_ratio_1 = edge_type_1 / edge_type_1.sum()
    edge_type_ratio_2 = edge_type_2 / edge_type_2.sum()
    edge_type_delta = edge_type_ratio_2 - edge_type_ratio_1
    top_edge_type_shift = np.argsort(-np.abs(edge_type_delta))[:3]

    high_drift = drift_rows[:5]
    summary_lines = [
        "# phase1 -> phase2 漂移摘要",
        "",
        "## 高漂移特征",
        "",
    ]
    for row in high_drift:
        summary_lines.append(
            f"- {row['feature_name']}: PSI={row['psi']:.4f}, "
            f"mean_shift={row['mean_shift']:.4f}, "
            f"neg1_ratio {row['phase1_neg1_ratio']:.4f}->{row['phase2_neg1_ratio']:.4f}"
        )
    summary_lines.extend(
        [
            "",
            "## 结构与时间差异",
            "",
            (
                f"- 总度均值: {np.mean(total_degree_1):.4f} -> {np.mean(total_degree_2):.4f}; "
                f"95分位: {np.quantile(total_degree_1, 0.95):.4f} -> "
                f"{np.quantile(total_degree_2, 0.95):.4f}"
            ),
            (
                f"- 活跃时间跨度: phase1={temporal_1['edge_count_by_day'].size} 天, "
                f"phase2={temporal_2['edge_count_by_day'].size} 天"
            ),
            (
                f"- 每日边数均值: {np.mean(temporal_1['edge_count_by_day']):.2f} -> "
                f"{np.mean(temporal_2['edge_count_by_day']):.2f}"
            ),
            "",
            "## 边类型份额变化最大的类型",
            "",
        ]
    )
    for idx in top_edge_type_shift.tolist():
        summary_lines.append(
            f"- type_{idx + 1}: {edge_type_ratio_1[idx]:.4f} -> {edge_type_ratio_2[idx]:.4f} "
            f"(delta={edge_type_delta[idx]:+.4f})"
        )
    summary_lines.append("")
    summary_lines.append("## 结论")
    summary_lines.append(
        "- phase2 在时间跨度、边类型构成和若干高缺失特征上均出现明显漂移，后续模型应显式保留时间编码，并把 phase2 视为外部泛化评估集。"
    )

    summary_text = "\n".join(summary_lines)
    (outdir / "drift_report.md").write_text(summary_text, encoding="utf-8")
    return drift_rows, summary_text


def build_recommended_split(
    outdir: Path,
    temporal_core: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    train_artifact = split_train_artifact()
    val_artifact = split_val_artifact()
    graph = load_phase(train_artifact)
    if temporal_core is None:
        temporal_core = compute_temporal_core(graph)
    first_active = temporal_core["first_active"]
    train_ids = graph.train_mask
    train_first_active = first_active[train_ids]

    candidate_quantiles = (0.80, 0.75, 0.70, 0.65, 0.60, 0.55)
    split_threshold = None
    split_train_ids = train_ids
    split_val_ids = np.empty(0, dtype=np.int32)
    for quantile in candidate_quantiles:
        threshold = int(np.quantile(train_first_active, quantile))
        val_mask = train_first_active >= threshold
        candidate_val_ids = train_ids[val_mask]
        candidate_train_ids = train_ids[~val_mask]
        val_labels = graph.y[candidate_val_ids]
        if candidate_val_ids.size == 0 or candidate_train_ids.size == 0:
            continue
        if np.any(val_labels == 0) and np.any(val_labels == 1):
            split_threshold = threshold
            split_train_ids = candidate_train_ids
            split_val_ids = candidate_val_ids
            break

    if split_threshold is None:
        raise RuntimeError("Could not build a validation split with both classes present.")

    train_first_median = float(np.median(first_active[split_train_ids]))
    val_first_median = float(np.median(first_active[split_val_ids]))
    if not val_first_median > train_first_median:
        raise RuntimeError(
            "Recommended split is not time-aware enough: validation median is not later."
        )

    np.save(outdir / "train_ids.npy", split_train_ids)
    np.save(outdir / "val_ids.npy", split_val_ids)

    split_summary: dict[str, Any] = {
        "split_style": str(ACTIVE_DATASET_SPEC.split_style),
        "train_phase": train_artifact,
        "val_phase": val_artifact,
        "external_phase": split_external_artifact(),
        "threshold_day": int(split_threshold),
        "train_split": {
            "threshold_day": int(split_threshold),
            "size": int(split_train_ids.size),
            "positive_count": int(np.sum(graph.y[split_train_ids] == 1)),
            "positive_rate": float(np.mean(graph.y[split_train_ids] == 1)),
            "first_active_median": train_first_median,
            "id_path": "train_ids.npy",
        },
        "val_split": {
            "threshold_day": int(split_threshold),
            "size": int(split_val_ids.size),
            "positive_count": int(np.sum(graph.y[split_val_ids] == 1)),
            "positive_rate": float(np.mean(graph.y[split_val_ids] == 1)),
            "first_active_median": val_first_median,
            "id_path": "val_ids.npy",
        },
    }

    if graph.test_mask.size:
        np.save(outdir / "test_pool_ids.npy", graph.test_mask)
        split_summary["test_pool"] = {
            "size": int(graph.test_mask.size),
            "id_path": "test_pool_ids.npy",
        }
        # Keep the legacy alias so older readers do not break immediately.
        split_summary["unlabeled_pool"] = {
            "size": int(graph.test_mask.size),
            "id_path": "test_pool_ids.npy",
        }

    external_artifact = split_external_artifact()
    if external_artifact:
        try:
            phase2 = load_phase(external_artifact)
        except FileNotFoundError:
            phase2 = None
        if phase2 is not None:
            np.save(outdir / "external_eval_ids.npy", phase2.train_mask)
            split_summary["external_eval"] = {
                "size": int(phase2.train_mask.size),
                "positive_count": int(np.sum(phase2.y[phase2.train_mask] == 1)),
                "positive_rate": float(np.mean(phase2.y[phase2.train_mask] == 1)),
                "id_path": "external_eval_ids.npy",
            }

    write_json(outdir / "recommended_split.json", split_summary)
    return split_summary


def run_analysis(phases: list[str], analyses: list[str], outdir: Path) -> dict[str, Any]:
    configure_matplotlib()
    ensure_dir(outdir)

    if "all" in analyses:
        analyses = list(ALL_ANALYSES)

    dataset_summary: dict[str, Any] = {"phases": {}}
    feature_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    phase_temporal_cache: dict[str, dict[str, np.ndarray]] = {}

    for phase in phases:
        data = load_phase(phase)
        phase_summary: dict[str, Any] = {
            "dataset_path": str(data.path),
        }
        temporal_core = None
        if "feature" in analyses or "temporal" in analyses or ("split" in analyses and phase == split_train_artifact()):
            temporal_core = compute_temporal_core(data)
        if "overview" in analyses:
            phase_summary["overview"] = analyze_overview(data, outdir)
        if "feature" in analyses:
            rows, compact = analyze_features(data, outdir, temporal_core=temporal_core)
            feature_rows.extend(rows)
            phase_summary["feature"] = compact
        if "graph" in analyses:
            rows, compact = analyze_graph(data, outdir)
            graph_rows.extend(rows)
            phase_summary["graph"] = compact
        if "temporal" in analyses or ("split" in analyses and phase == split_train_artifact()):
            if temporal_core is None:
                temporal_core = compute_temporal_core(data)
            temporal_rows_phase, compact, temporal_core = analyze_temporal(
                data, outdir, temporal_core=temporal_core
            )
            phase_temporal_cache[phase] = temporal_core
            if "temporal" in analyses:
                temporal_rows.extend(temporal_rows_phase)
                phase_summary["temporal"] = compact
        dataset_summary["phases"][phase] = phase_summary

    if feature_rows:
        write_csv(outdir / "feature_profile.csv", feature_rows)
    if graph_rows:
        write_csv(outdir / "graph_profile.csv", graph_rows)
    if temporal_rows:
        write_csv(outdir / "temporal_profile.csv", temporal_rows)

    if "drift" in analyses and ACTIVE_DATASET_SPEC.split_style == "two_phase":
        drift_rows, _ = analyze_drift(outdir)
        dataset_summary["drift"] = {
            "top_feature_drift": drift_rows[:5],
            "feature_drift_path": "feature_drift.csv",
            "drift_report_path": "drift_report.md",
        }

    if "split" in analyses:
        split_summary = build_recommended_split(
            outdir, temporal_core=phase_temporal_cache.get(split_train_artifact())
        )
        dataset_summary["recommended_split"] = split_summary

    write_json(outdir / "analysis_overview.json", dataset_summary)
    return dataset_summary
