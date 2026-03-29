from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiment.training.common import ensure_dir, write_json
from experiment.training.features import FeatureStore, default_feature_groups


DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "n_estimators": 1500,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 0,
    "n_jobs": -1,
}


class LightGBMExperiment:
    def __init__(
        self,
        model_name: str,
        seed: int,
        feature_groups: list[str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.feature_groups = feature_groups or default_feature_groups(model_name)
        self.params = dict(DEFAULT_LGB_PARAMS)
        if params:
            self.params.update(params)
        self.params["random_state"] = seed
        self.booster = None
        self.best_iteration = None

    def fit(
        self,
        train_store: FeatureStore,
        train_ids: np.ndarray,
        train_labels: np.ndarray,
        val_ids: np.ndarray,
        val_labels: np.ndarray,
    ) -> dict[str, float]:
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score

        x_train = train_store.take_rows(train_ids)
        x_val = train_store.take_rows(val_ids)
        pos_count = float(np.sum(train_labels == 1))
        neg_count = float(np.sum(train_labels == 0))
        scale_pos_weight = neg_count / max(pos_count, 1.0)

        model = lgb.LGBMClassifier(
            **self.params,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(
            x_train,
            train_labels,
            eval_set=[(x_val, val_labels)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )
        self.booster = model.booster_
        self.best_iteration = int(self.booster.best_iteration or self.params["n_estimators"])
        val_prob = self.predict_proba(train_store, val_ids)
        val_auc = float(roc_auc_score(val_labels, val_prob))
        return {
            "val_auc": val_auc,
            "best_iteration": float(self.best_iteration),
        }

    def predict_proba(self, feature_store: FeatureStore, node_ids: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("LightGBM booster is not fitted.")
        x = feature_store.take_rows(node_ids)
        pred = self.booster.predict(x, num_iteration=self.best_iteration)
        return np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)

    def save(self, run_dir: Path, feature_names: list[str]) -> None:
        if self.booster is None:
            raise RuntimeError("LightGBM booster is not fitted.")
        ensure_dir(run_dir)
        model_path = run_dir / "model.txt"
        model_text = self.booster.model_to_string(num_iteration=self.best_iteration)
        model_path.write_text(model_text, encoding="utf-8")
        metadata = {
            "model_name": self.model_name,
            "seed": self.seed,
            "feature_groups": self.feature_groups,
            "best_iteration": self.best_iteration,
            "feature_names": feature_names,
        }
        write_json(run_dir / "model_meta.json", metadata)
        self._write_feature_importance(run_dir / "feature_importance.csv", feature_names)

    def _write_feature_importance(self, path: Path, feature_names: list[str]) -> None:
        if self.booster is None:
            raise RuntimeError("LightGBM booster is not fitted.")
        gain = self.booster.feature_importance(importance_type="gain")
        split = self.booster.feature_importance(importance_type="split")
        rows = [
            {
                "feature_name": name,
                "gain": float(g),
                "split": int(s),
            }
            for name, g, s in zip(feature_names, gain.tolist(), split.tolist(), strict=True)
        ]
        rows.sort(key=lambda row: row["gain"], reverse=True)
        with path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["feature_name", "gain", "split"])
            writer.writeheader()
            writer.writerows(rows)

    @classmethod
    def load(cls, run_dir: Path) -> "LightGBMExperiment":
        import lightgbm as lgb

        meta = json.loads((run_dir / "model_meta.json").read_text(encoding="utf-8"))
        instance = cls(
            model_name=meta["model_name"],
            seed=int(meta["seed"]),
            feature_groups=list(meta["feature_groups"]),
        )
        instance.best_iteration = int(meta["best_iteration"])
        instance.booster = lgb.Booster(model_str=(run_dir / "model.txt").read_text(encoding="utf-8"))
        return instance
