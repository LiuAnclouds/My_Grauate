from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (SRC_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dyrift.data_processing.core.contracts import save_prepared_phase
from dyrift.data_processing.core.elliptic import (
    build_chronological_node_contracts,
    build_edge_arrays,
    build_full_graph_contract,
    load_pandas,
    map_elliptic_binary_labels,
    phase_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a full Elliptic++ prepared graph without downloading the large "
            "txs_features.csv. The first 166 feature columns are reconstructed from "
            "the complete original Elliptic transaction feature table, which has the "
            "same txId/time/local/aggregate feature contract. Existing Elliptic++ "
            "transaction-stat columns are kept where present. Missing rows can be "
            "marked as -1, filled with simple statistics, or imputed/recovered "
            "from the observed Elliptic++ rows without using labels."
        )
    )
    parser.add_argument(
        "--elliptic-features",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "elliptic_transactions" / "raw" / "elliptic_txs_features.csv",
        help="Complete original Elliptic feature CSV, no header.",
    )
    parser.add_argument(
        "--ellipticpp-classes",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "ellipticpp_transactions" / "raw" / "txs_classes.csv",
        help="Elliptic++ transaction classes CSV.",
    )
    parser.add_argument(
        "--ellipticpp-edgelist",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "ellipticpp_transactions" / "raw" / "txs_edgelist.csv",
        help="Elliptic++ transaction edgelist CSV.",
    )
    parser.add_argument(
        "--ellipticpp-partial-features",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "ellipticpp_transactions" / "raw" / "txs_features.csv",
        help="Local Elliptic++ txs_features.csv, allowed to be truncated.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "ellipticpp_transactions" / "prepared",
        help="Directory where graph_gdata.npz, phase1_gdata.npz, and phase2_gdata.npz are written.",
    )
    parser.add_argument(
        "--phase1-max-step",
        type=int,
        default=34,
        help="Last time step included in phase1.",
    )
    parser.add_argument(
        "--extra-fill-strategy",
        choices=("missing", "median", "zero", "drop", "learned", "linear_map"),
        default="missing",
        help=(
            "How to handle Elliptic++-only transaction-stat columns for rows missing "
            "from the truncated local txs_features.csv. `missing` writes -1 and lets "
            "UTPM missing-mask logic handle them. `median` uses observed-column "
            "medians. `zero` writes zeros. `linear_map` learns label-free affine "
            "mappings from the anonymized base feature columns back to the public "
            "Elliptic++ transaction-stat columns. `learned` fits label-free ridge imputers "
            "from observed rows using the Elliptic base features plus graph degrees. "
            "`drop` omits these 17 columns and builds a 166-column raw contract."
        ),
    )
    parser.add_argument(
        "--learned-extra-alpha",
        type=float,
        default=10.0,
        help="Ridge alpha for --extra-fill-strategy learned.",
    )
    parser.add_argument(
        "--learned-extra-sample-limit",
        type=int,
        default=0,
        help=(
            "Optional maximum observed rows per learned extra-column imputer. "
            "0 means use all observed rows."
        ),
    )
    parser.add_argument(
        "--derive-degree-columns",
        action="store_true",
        help="Recompute in_txs_degree and out_txs_degree from the full tx-tx graph for every node.",
    )
    parser.add_argument(
        "--linear-map-target-train-missing-count-mean",
        type=float,
        default=0.0,
        help=(
            "If > 0, keep a small deterministic subset of training rows masked as -1 "
            "after linear-map reconstruction so the mean UTPM missing-count on the train split "
            "approaches the requested value."
        ),
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "analysis" / "ellipticpp_transactions",
        help="Directory containing train_ids.npy and val_ids.npy for the active split.",
    )
    return parser.parse_args()


def _load_elliptic_features(path: Path):
    pd = load_pandas()
    feature_col_count = int(pd.read_csv(path, header=None, nrows=1).shape[1])
    if feature_col_count < 3:
        raise RuntimeError(f"Unexpected Elliptic feature width: {feature_col_count}")
    feature_dtype = {0: "int64", 1: "int16"}
    feature_dtype.update({idx: "float32" for idx in range(2, feature_col_count)})
    return pd.read_csv(path, header=None, dtype=feature_dtype, low_memory=False)


def _load_partial_epp_extra_features(path: Path, expected_base_width: int) -> tuple[dict[int, np.ndarray], list[str]]:
    pd = load_pandas()
    if not path.exists() or path.stat().st_size == 0:
        return {}, []
    header = list(pd.read_csv(path, nrows=0).columns)
    if "txId" not in header or "Time step" not in header:
        raise RuntimeError(f"Unexpected Elliptic++ feature header in {path}")
    # expected_base_width includes Time step + the 165 local/aggregate features.
    extra_columns = header[1 + int(expected_base_width) :]
    if not extra_columns:
        return {}, []
    usecols = ["txId", *extra_columns]
    dtype = {"txId": "int64", **{column: "float32" for column in extra_columns}}
    partial = pd.read_csv(path, usecols=usecols, dtype=dtype, low_memory=False)
    extra_by_txid: dict[int, np.ndarray] = {}
    for row in partial.itertuples(index=False):
        tx_id = int(row[0])
        values = np.asarray(row[1:], dtype=np.float32)
        extra_by_txid[tx_id] = values
    return extra_by_txid, [str(column) for column in extra_columns]


def _compute_degrees(edges, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(node_ids.tolist())}
    raw_src = edges["txId1"].to_numpy(dtype=np.int64, copy=False)
    raw_dst = edges["txId2"].to_numpy(dtype=np.int64, copy=False)
    src_idx = np.fromiter(
        (node_to_idx.get(int(tx_id), -1) for tx_id in raw_src),
        dtype=np.int32,
        count=raw_src.shape[0],
    )
    dst_idx = np.fromiter(
        (node_to_idx.get(int(tx_id), -1) for tx_id in raw_dst),
        dtype=np.int32,
        count=raw_dst.shape[0],
    )
    valid = (src_idx >= 0) & (dst_idx >= 0)
    out_degree = np.bincount(src_idx[valid], minlength=node_ids.shape[0]).astype(np.float32, copy=False)
    in_degree = np.bincount(dst_idx[valid], minlength=node_ids.shape[0]).astype(np.float32, copy=False)
    return in_degree, out_degree


def _build_observed_extra_matrix(
    *,
    node_ids: np.ndarray,
    extra_by_txid: dict[int, np.ndarray],
    extra_columns: list[str],
    fill_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    extra_x = np.tile(fill_values.reshape(1, -1), (node_ids.shape[0], 1)).astype(np.float32, copy=False)
    observed_mask = np.zeros(node_ids.shape[0], dtype=bool)
    covered = 0
    for row_idx, tx_id in enumerate(node_ids.tolist()):
        values = extra_by_txid.get(int(tx_id))
        if values is not None:
            if values.shape[0] != len(extra_columns):
                raise RuntimeError(
                    f"Unexpected extra feature width for txId={tx_id}: "
                    f"{values.shape[0]} != {len(extra_columns)}"
                )
            extra_x[row_idx, :] = np.where(np.isfinite(values), values, fill_values)
            observed_mask[row_idx] = True
            covered += 1
    return extra_x, observed_mask, covered


def _learned_extra_fill(
    *,
    base_x: np.ndarray,
    node_ids: np.ndarray,
    time_steps: np.ndarray,
    extra_by_txid: dict[int, np.ndarray],
    extra_columns: list[str],
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    sample_limit: int,
    alpha: float,
) -> tuple[np.ndarray, int, dict[str, object]]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "--extra-fill-strategy learned requires scikit-learn in the active environment."
        ) from exc

    observed_extra = np.stack(list(extra_by_txid.values()), axis=0).astype(np.float32, copy=False)
    fill_values = np.nanmedian(observed_extra, axis=0).astype(np.float32, copy=False)
    fill_values = np.nan_to_num(fill_values, nan=0.0, posinf=0.0, neginf=0.0)
    extra_x, observed_mask, covered = _build_observed_extra_matrix(
        node_ids=node_ids,
        extra_by_txid=extra_by_txid,
        extra_columns=extra_columns,
        fill_values=fill_values,
    )

    total_degree = (in_degree + out_degree).astype(np.float32, copy=False)
    max_time = float(max(int(np.max(time_steps)), 1))
    graph_predictors = np.column_stack(
        [
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(total_degree),
            time_steps.astype(np.float32, copy=False),
            time_steps.astype(np.float32, copy=False) / max_time,
        ]
    ).astype(np.float32, copy=False)
    predictors = np.concatenate([base_x, graph_predictors], axis=1).astype(np.float32, copy=False)

    observed_idx = np.flatnonzero(observed_mask)
    missing_idx = np.flatnonzero(~observed_mask)
    if observed_idx.size < 100:
        raise RuntimeError(
            f"Not enough observed Elliptic++ extra rows for learned fill: {observed_idx.size}"
        )

    observed_times = time_steps[observed_idx]
    holdout_floor = int(np.max(observed_times)) - 1
    valid_idx = observed_idx[observed_times >= holdout_floor]
    train_idx = observed_idx[observed_times < holdout_floor]
    if train_idx.size < 1000 or valid_idx.size < 100:
        rng = np.random.default_rng(17)
        shuffled = observed_idx.copy()
        rng.shuffle(shuffled)
        split = max(int(round(shuffled.size * 0.8)), 1)
        train_idx = shuffled[:split]
        valid_idx = shuffled[split:]

    rng = np.random.default_rng(17)
    if sample_limit and sample_limit > 0 and train_idx.size > int(sample_limit):
        train_idx = rng.choice(train_idx, size=int(sample_limit), replace=False)

    diagnostics: dict[str, object] = {
        "model": "ridge_log_target",
        "alpha": float(alpha),
        "observed_rows": int(observed_idx.size),
        "missing_rows_imputed": int(missing_idx.size),
        "train_rows": int(train_idx.size),
        "validation_rows": int(valid_idx.size),
        "validation_policy": "observed_latest_two_timesteps_holdout",
        "columns": {},
    }
    column_diagnostics: dict[str, dict[str, float | int | str]] = {}
    name_to_extra_idx = {name: idx for idx, name in enumerate(extra_columns)}
    exact_degree_columns = {"in_txs_degree", "out_txs_degree"}

    x_train = predictors[train_idx]
    x_valid = predictors[valid_idx] if valid_idx.size else None
    x_missing = predictors[missing_idx] if missing_idx.size else None

    for column_idx, column_name in enumerate(extra_columns):
        if column_name in exact_degree_columns:
            column_diagnostics[column_name] = {"fill": "exact_graph_degree"}
            continue

        observed_column = extra_x[observed_idx, column_idx].astype(np.float32, copy=False)
        observed_column = observed_column[np.isfinite(observed_column)]
        if observed_column.size == 0:
            column_diagnostics[column_name] = {"fill": "fallback_zero_no_finite_observed"}
            extra_x[missing_idx, column_idx] = 0.0
            continue
        finite_train_idx = train_idx[np.isfinite(extra_x[train_idx, column_idx])]
        finite_valid_idx = valid_idx[np.isfinite(extra_x[valid_idx, column_idx])] if valid_idx.size else valid_idx
        if finite_train_idx.size < 100:
            column_diagnostics[column_name] = {"fill": "fallback_median_too_few_finite_observed"}
            extra_x[missing_idx, column_idx] = float(np.nanmedian(observed_column))
            continue
        train_y = extra_x[finite_train_idx, column_idx].astype(np.float32, copy=False)
        valid_y = (
            extra_x[finite_valid_idx, column_idx].astype(np.float32, copy=False)
            if finite_valid_idx.size
            else None
        )
        nonnegative = bool(np.nanmin(observed_column) >= 0.0)
        if nonnegative:
            fit_y = np.log1p(np.clip(train_y, 0.0, None))
        else:
            fit_y = train_y

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=max(float(alpha), 1e-6), random_state=17),
        )
        model.fit(predictors[finite_train_idx], fit_y)

        valid_mae = valid_r2 = None
        if valid_y is not None and finite_valid_idx.size:
            valid_pred = model.predict(predictors[finite_valid_idx]).astype(np.float32, copy=False)
            if nonnegative:
                valid_pred = np.expm1(valid_pred)
            valid_pred = np.nan_to_num(valid_pred, nan=float(np.median(observed_column)))
            valid_mae = float(mean_absolute_error(valid_y, valid_pred))
            try:
                valid_r2 = float(r2_score(valid_y, valid_pred))
            except ValueError:
                valid_r2 = None

        if x_missing is not None and missing_idx.size:
            pred = model.predict(x_missing).astype(np.float32, copy=False)
            if nonnegative:
                pred = np.expm1(pred)
            lo = float(np.quantile(observed_column, 0.001))
            hi = float(np.quantile(observed_column, 0.999))
            if not np.isfinite(lo):
                lo = float(np.nanmin(observed_column))
            if not np.isfinite(hi):
                hi = float(np.nanmax(observed_column))
            if hi < lo:
                lo, hi = hi, lo
            pred = np.nan_to_num(pred, nan=float(np.median(observed_column)), posinf=hi, neginf=lo)
            pred = np.clip(pred, lo, hi)
            extra_x[missing_idx, column_idx] = pred.astype(np.float32, copy=False)

        column_diagnostics[column_name] = {
            "fill": "ridge_log_target" if nonnegative else "ridge_raw_target",
            "observed_min": float(np.nanmin(observed_column)),
            "observed_median": float(np.nanmedian(observed_column)),
            "observed_max": float(np.nanmax(observed_column)),
            "validation_mae": -1.0 if valid_mae is None else float(valid_mae),
            "validation_r2": -999.0 if valid_r2 is None else float(valid_r2),
        }

    if "in_txs_degree" in name_to_extra_idx:
        extra_x[:, name_to_extra_idx["in_txs_degree"]] = in_degree
    if "out_txs_degree" in name_to_extra_idx:
        extra_x[:, name_to_extra_idx["out_txs_degree"]] = out_degree
    diagnostics["columns"] = column_diagnostics
    return extra_x.astype(np.float32, copy=False), covered, diagnostics


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return -999.0
    denom = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if denom <= 1e-12:
        return 1.0 if float(np.max(np.abs(y_true - y_pred))) <= 1e-8 else -999.0
    return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))


def _mean_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return -1.0
    return float(np.mean(np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))))


def _linear_lstsq_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.concatenate(
        [np.asarray(x, dtype=np.float64), np.ones((x.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    coeff, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64), rcond=None)
    return coeff[:-1], float(coeff[-1])


def _linear_lstsq_predict(x: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) @ np.asarray(weights, dtype=np.float64) + float(bias))


def _linear_map_extra_fill(
    *,
    base_x: np.ndarray,
    node_ids: np.ndarray,
    time_steps: np.ndarray,
    extra_by_txid: dict[int, np.ndarray],
    extra_columns: list[str],
    in_degree: np.ndarray,
    out_degree: np.ndarray,
) -> tuple[np.ndarray, int, dict[str, object]]:
    observed_extra = np.stack(list(extra_by_txid.values()), axis=0).astype(np.float32, copy=False)
    fill_values = np.nanmedian(observed_extra, axis=0).astype(np.float32, copy=False)
    fill_values = np.nan_to_num(fill_values, nan=0.0, posinf=0.0, neginf=0.0)
    extra_x, observed_mask, covered = _build_observed_extra_matrix(
        node_ids=node_ids,
        extra_by_txid=extra_by_txid,
        extra_columns=extra_columns,
        fill_values=fill_values,
    )

    name_to_extra_idx = {name: idx for idx, name in enumerate(extra_columns)}
    if "in_txs_degree" in name_to_extra_idx:
        extra_x[:, name_to_extra_idx["in_txs_degree"]] = in_degree
    if "out_txs_degree" in name_to_extra_idx:
        extra_x[:, name_to_extra_idx["out_txs_degree"]] = out_degree

    observed_idx = np.flatnonzero(observed_mask)
    missing_idx = np.flatnonzero(~observed_mask)
    observed_times = time_steps[observed_idx]
    latest_observed_time = int(np.max(observed_times)) if observed_idx.size else -1
    valid_idx = observed_idx[observed_times >= latest_observed_time]
    train_idx = observed_idx[observed_times < latest_observed_time]
    if train_idx.size < 1000 or valid_idx.size < 100:
        rng = np.random.default_rng(23)
        shuffled = observed_idx.copy()
        rng.shuffle(shuffled)
        split = max(int(round(shuffled.size * 0.8)), 1)
        train_idx = shuffled[:split]
        valid_idx = shuffled[split:]

    diagnostics: dict[str, object] = {
        "model": "affine_base_feature_map",
        "observed_rows": int(observed_idx.size),
        "missing_rows_imputed": int(missing_idx.size),
        "train_rows": int(train_idx.size),
        "validation_rows": int(valid_idx.size),
        "validation_policy": "latest_observed_timestep_holdout",
        "columns": {},
    }
    column_diagnostics: dict[str, dict[str, object]] = {}
    exact_degree_columns = {"in_txs_degree", "out_txs_degree"}
    integer_columns = {
        "in_txs_degree",
        "out_txs_degree",
        "size",
        "num_input_addresses",
        "num_output_addresses",
    }

    base_x64 = np.asarray(base_x, dtype=np.float64)
    k_candidates = (1, 2, 3, 5, 8, 12, 20)

    for column_idx, column_name in enumerate(extra_columns):
        if column_name in exact_degree_columns:
            column_diagnostics[column_name] = {
                "fill": "exact_graph_degree",
                "validation_r2": 1.0,
                "validation_mae": 0.0,
            }
            continue

        train_finite = train_idx[np.isfinite(extra_x[train_idx, column_idx])]
        valid_finite = valid_idx[np.isfinite(extra_x[valid_idx, column_idx])]
        all_finite = observed_idx[np.isfinite(extra_x[observed_idx, column_idx])]
        if train_finite.size < 100 or all_finite.size < 100:
            fallback = float(np.nanmedian(extra_x[all_finite, column_idx])) if all_finite.size else 0.0
            extra_x[missing_idx, column_idx] = fallback
            column_diagnostics[column_name] = {
                "fill": "fallback_median_too_few_finite_observed",
                "validation_r2": -999.0,
                "validation_mae": -1.0,
            }
            continue

        y_train = extra_x[train_finite, column_idx].astype(np.float64, copy=False)
        x_train = base_x64[train_finite]
        centered_x = x_train - np.mean(x_train, axis=0, keepdims=True)
        centered_y = y_train - float(np.mean(y_train))
        denom = (
            np.sqrt(np.sum(centered_x * centered_x, axis=0))
            * max(float(np.sqrt(np.sum(centered_y * centered_y))), 1e-12)
            + 1e-12
        )
        corr = (centered_x.T @ centered_y) / denom
        ranked_features = np.argsort(np.abs(corr))[::-1]

        best: dict[str, object] | None = None
        for k in k_candidates:
            selected = ranked_features[: min(int(k), ranked_features.shape[0])]
            weights, bias = _linear_lstsq_fit(base_x64[train_finite][:, selected], y_train)
            if valid_finite.size:
                valid_pred = _linear_lstsq_predict(base_x64[valid_finite][:, selected], weights, bias)
                valid_true = extra_x[valid_finite, column_idx].astype(np.float64, copy=False)
                valid_r2 = _safe_r2(valid_true, valid_pred)
                valid_mae = _mean_abs_error(valid_true, valid_pred)
            else:
                valid_r2 = -999.0
                valid_mae = -1.0
            candidate = {
                "selected": selected,
                "weights": weights,
                "bias": bias,
                "validation_r2": float(valid_r2),
                "validation_mae": float(valid_mae),
            }
            if best is None:
                best = candidate
            else:
                best_r2 = float(best["validation_r2"])
                best_mae = float(best["validation_mae"])
                if (valid_r2 > best_r2 + 1e-9) or (
                    abs(valid_r2 - best_r2) <= 1e-9 and valid_mae >= 0.0 and valid_mae < best_mae
                ):
                    best = candidate

        assert best is not None
        selected = np.asarray(best["selected"], dtype=np.int64)
        weights, bias = _linear_lstsq_fit(base_x64[all_finite][:, selected], extra_x[all_finite, column_idx])
        if missing_idx.size:
            pred = _linear_lstsq_predict(base_x64[missing_idx][:, selected], weights, bias)
            observed_column = extra_x[all_finite, column_idx].astype(np.float64, copy=False)
            if float(np.nanmin(observed_column)) >= 0.0:
                pred = np.clip(pred, 0.0, None)
            if column_name in integer_columns:
                pred = np.rint(pred)
                pred = np.clip(pred, 0.0, None)
            extra_x[missing_idx, column_idx] = pred.astype(np.float32, copy=False)

        column_diagnostics[column_name] = {
            "fill": "affine_base_feature_map",
            "selected_base_feature_indices": [int(v) for v in selected.tolist()],
            "validation_r2": float(best["validation_r2"]),
            "validation_mae": float(best["validation_mae"]),
            "observed_min": float(np.nanmin(extra_x[all_finite, column_idx])),
            "observed_median": float(np.nanmedian(extra_x[all_finite, column_idx])),
            "observed_max": float(np.nanmax(extra_x[all_finite, column_idx])),
        }

    diagnostics["columns"] = column_diagnostics
    return extra_x.astype(np.float32, copy=False), covered, diagnostics


def _apply_target_train_missing_mask(
    *,
    extra_x: np.ndarray,
    base_x: np.ndarray,
    train_ids: np.ndarray,
    observed_mask: np.ndarray,
    extra_columns: list[str],
    in_degree: np.ndarray,
    out_degree: np.ndarray,
    target_train_missing_count_mean: float,
    preserve_degree_columns: bool,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "target_train_missing_count_mean": float(target_train_missing_count_mean),
        "selected_train_rows": 0,
        "candidate_train_rows": 0,
        "effective_missing_columns": 0,
        "selection_policy": "train_missing_candidates_by_low_extra_l1",
    }
    if target_train_missing_count_mean <= 0.0 or train_ids.size == 0:
        diagnostics["selection_policy"] = "disabled"
        return diagnostics

    exact_degree_columns = {"in_txs_degree", "out_txs_degree"} if preserve_degree_columns else set()
    effective_missing_columns = len(extra_columns) - len([name for name in exact_degree_columns if name in extra_columns])
    effective_missing_columns = max(int(effective_missing_columns), 1)
    target_rows = int(round(float(target_train_missing_count_mean) * float(train_ids.size) / float(effective_missing_columns)))
    target_rows = max(target_rows, 0)
    diagnostics["effective_missing_columns"] = int(effective_missing_columns)
    diagnostics["target_rows"] = int(target_rows)

    if target_rows == 0:
        diagnostics["selection_policy"] = "disabled_rounding_to_zero"
        return diagnostics

    train_ids = np.asarray(train_ids, dtype=np.int64)
    candidate_mask = ~observed_mask[train_ids]
    candidate_ids = train_ids[candidate_mask]
    if candidate_ids.size == 0:
        candidate_ids = train_ids
        diagnostics["selection_policy"] = "train_rows_fallback_no_missing_candidates"
    diagnostics["candidate_train_rows"] = int(candidate_ids.size)

    candidate_scores = np.sum(np.abs(np.asarray(extra_x[candidate_ids], dtype=np.float64)), axis=1)
    selected_count = min(target_rows, candidate_ids.size)
    if selected_count <= 0:
        return diagnostics
    selected = candidate_ids[np.argpartition(candidate_scores, selected_count - 1)[:selected_count]]
    extra_x[selected, :] = -1.0

    if preserve_degree_columns:
        name_to_extra_idx = {name: idx for idx, name in enumerate(extra_columns)}
        if "in_txs_degree" in name_to_extra_idx:
            extra_x[:, name_to_extra_idx["in_txs_degree"]] = in_degree
        if "out_txs_degree" in name_to_extra_idx:
            extra_x[:, name_to_extra_idx["out_txs_degree"]] = out_degree

    diagnostics["selected_train_rows"] = int(selected.size)
    diagnostics["selection_policy"] = (
        "train_missing_candidates_by_low_extra_l1"
        if candidate_mask.any()
        else "train_rows_by_low_extra_l1"
    )
    diagnostics["selected_train_row_ids_preview"] = [int(v) for v in selected[:20].tolist()]
    return diagnostics


def main() -> None:
    args = parse_args()
    pd = load_pandas()

    elliptic_features = _load_elliptic_features(args.elliptic_features)
    node_ids = elliptic_features.iloc[:, 0].to_numpy(dtype=np.int64, copy=False)
    time_steps = elliptic_features.iloc[:, 1].to_numpy(dtype=np.int32, copy=False)
    base_x = elliptic_features.iloc[:, 1:].to_numpy(dtype=np.float32, copy=True)

    extra_by_txid, extra_columns = _load_partial_epp_extra_features(
        args.ellipticpp_partial_features,
        expected_base_width=int(base_x.shape[1]),
    )
    classes = pd.read_csv(args.ellipticpp_classes, dtype={"txId": "int64", "class": "string"})
    class_series = classes.set_index("txId").reindex(node_ids)["class"]
    if class_series.isna().any():
        missing = int(class_series.isna().sum())
        raise RuntimeError(f"Missing class labels for {missing} transactions from Elliptic++ classes.")
    labels = map_elliptic_binary_labels(class_series)

    edges = pd.read_csv(args.ellipticpp_edgelist, dtype={"txId1": "int64", "txId2": "int64"})
    edge_index, edge_timestamp = build_edge_arrays(edges, node_ids=node_ids, time_steps=time_steps)
    in_degree, out_degree = _compute_degrees(edges, node_ids)
    train_ids = np.load(args.analysis_root / "train_ids.npy") if args.analysis_root.exists() else np.empty(0, dtype=np.int32)

    learned_extra_diagnostics: dict[str, object] | None = None
    mask_diagnostics: dict[str, object] | None = None
    if args.extra_fill_strategy == "drop":
        x = base_x
        extra_x = np.zeros((node_ids.shape[0], 0), dtype=np.float32)
        covered = 0
        extra_columns = []
    elif extra_columns:
        if args.extra_fill_strategy == "linear_map":
            extra_x, covered, learned_extra_diagnostics = _linear_map_extra_fill(
                base_x=base_x,
                node_ids=node_ids,
                time_steps=time_steps,
                extra_by_txid=extra_by_txid,
                extra_columns=extra_columns,
                in_degree=in_degree,
                out_degree=out_degree,
            )
        elif args.extra_fill_strategy == "learned":
            extra_x, covered, learned_extra_diagnostics = _learned_extra_fill(
                base_x=base_x,
                node_ids=node_ids,
                time_steps=time_steps,
                extra_by_txid=extra_by_txid,
                extra_columns=extra_columns,
                in_degree=in_degree,
                out_degree=out_degree,
                sample_limit=int(args.learned_extra_sample_limit),
                alpha=float(args.learned_extra_alpha),
            )
        elif args.extra_fill_strategy == "median":
            observed_extra = np.stack(list(extra_by_txid.values()), axis=0).astype(np.float32, copy=False)
            fill_values = np.median(observed_extra, axis=0).astype(np.float32, copy=False)
            extra_x, observed_mask, covered = _build_observed_extra_matrix(
                node_ids=node_ids,
                extra_by_txid=extra_by_txid,
                extra_columns=extra_columns,
                fill_values=fill_values,
            )
        elif args.extra_fill_strategy == "zero":
            fill_values = np.zeros(len(extra_columns), dtype=np.float32)
            extra_x, observed_mask, covered = _build_observed_extra_matrix(
                node_ids=node_ids,
                extra_by_txid=extra_by_txid,
                extra_columns=extra_columns,
                fill_values=fill_values,
            )
        else:
            fill_values = np.full(len(extra_columns), -1.0, dtype=np.float32)
            extra_x, observed_mask, covered = _build_observed_extra_matrix(
                node_ids=node_ids,
                extra_by_txid=extra_by_txid,
                extra_columns=extra_columns,
                fill_values=fill_values,
            )

        if args.derive_degree_columns:
            name_to_extra_idx = {name: idx for idx, name in enumerate(extra_columns)}
            if "in_txs_degree" in name_to_extra_idx:
                extra_x[:, name_to_extra_idx["in_txs_degree"]] = in_degree
            if "out_txs_degree" in name_to_extra_idx:
                extra_x[:, name_to_extra_idx["out_txs_degree"]] = out_degree

        if args.extra_fill_strategy == "linear_map" and args.linear_map_target_train_missing_count_mean > 0.0:
            observed_mask = np.zeros(node_ids.shape[0], dtype=bool)
            if extra_by_txid:
                node_to_idx = {int(tx_id): idx for idx, tx_id in enumerate(node_ids.tolist())}
                for tx_id in extra_by_txid:
                    row_idx = node_to_idx.get(int(tx_id))
                    if row_idx is not None:
                        observed_mask[row_idx] = True
            mask_diagnostics = _apply_target_train_missing_mask(
                extra_x=extra_x,
                base_x=base_x,
                train_ids=train_ids,
                observed_mask=observed_mask,
                extra_columns=extra_columns,
                in_degree=in_degree,
                out_degree=out_degree,
                target_train_missing_count_mean=float(args.linear_map_target_train_missing_count_mean),
                preserve_degree_columns=bool(args.derive_degree_columns),
            )

        x = np.concatenate([base_x, extra_x], axis=1).astype(np.float32, copy=False)
    else:
        extra_x = np.zeros((node_ids.shape[0], 0), dtype=np.float32)
        covered = 0
        x = base_x

    graph = build_full_graph_contract(
        x=x,
        y=labels,
        edge_index=edge_index,
        edge_timestamp=edge_timestamp,
    )
    phase1, phase2 = build_chronological_node_contracts(
        x=x,
        y=labels,
        time_steps=time_steps,
        edge_index=edge_index,
        edge_timestamp=edge_timestamp,
        phase1_max_step=int(args.phase1_max_step),
    )

    args.prepared_dir.mkdir(parents=True, exist_ok=True)
    save_prepared_phase(args.prepared_dir / "graph_gdata.npz", graph)
    save_prepared_phase(args.prepared_dir / "phase1_gdata.npz", phase1)
    save_prepared_phase(args.prepared_dir / "phase2_gdata.npz", phase2)

    metadata = {
        "dataset": "ellipticpp_transactions",
        "rebuild_method": "elliptic_base_features_plus_partial_ellipticpp_extra_stats",
        "elliptic_features": str(args.elliptic_features),
        "ellipticpp_classes": str(args.ellipticpp_classes),
        "ellipticpp_edgelist": str(args.ellipticpp_edgelist),
        "ellipticpp_partial_features": str(args.ellipticpp_partial_features),
        "class_mapping": {
            "1": "illicit -> 1",
            "2": "licit -> 0",
            "3": "unknown -> -100",
        },
        "phase1_max_step": int(args.phase1_max_step),
        "feature_columns": {
            "base_from_elliptic": int(base_x.shape[1]),
            "extra_from_partial_ellipticpp": list(extra_columns),
            "extra_fill_strategy": str(args.extra_fill_strategy),
            "derive_degree_columns": bool(args.derive_degree_columns),
            "linear_map_target_train_missing_count_mean": float(args.linear_map_target_train_missing_count_mean),
            "extra_rows_covered": int(covered),
            "extra_rows_missing": int(node_ids.shape[0] - covered),
            "extra_rows_missing_marked_minus_one": (
                int(node_ids.shape[0] - covered)
                if args.extra_fill_strategy == "missing"
                else 0
            ),
            "learned_extra_diagnostics": learned_extra_diagnostics,
            "mask_diagnostics": mask_diagnostics,
        },
        "graph": phase_summary(graph),
        "phase2_full_history_graph": True,
        "phase1": phase_summary(phase1),
        "phase2": phase_summary(phase2),
    }
    (args.prepared_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: metadata[key] for key in ("feature_columns", "graph", "phase1", "phase2")}, indent=2))


if __name__ == "__main__":
    main()
