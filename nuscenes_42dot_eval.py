"""Custom NuScenes evaluation utilities for 42dot motion prediction pipeline."""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.config import PredictionConfig
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from dataclasses import dataclass


@dataclass
class PredictionMetrics:
    ade_k1: float   # Average Displacement Error (best of 1)
    ade_k5: float   # Average Displacement Error (best of 5)
    fde_k1: float   # Final Displacement Error (best of 1)
    fde_k5: float   # Final Displacement Error (best of 5)
    miss_rate: float


def compute_ade(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred: (K, T, 2), gt: (T, 2). Returns minADE over K modes."""
    errors = np.linalg.norm(pred - gt[None], axis=-1).mean(axis=-1)  # (K,)
    return float(errors.min())


def compute_fde(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred: (K, T, 2), gt: (T, 2). Returns minFDE over K modes."""
    errors = np.linalg.norm(pred[:, -1] - gt[-1], axis=-1)  # (K,)
    return float(errors.min())


class FortyTwoDotEvaluator:
    """Evaluator wrapping NuScenes prediction challenge for 42dot internal models."""

    def __init__(self, nusc: NuScenes, split: str = "val", config_name: str = "predict_2020_icra.json"):
        self.helper = PredictHelper(nusc)
        self.tokens = get_prediction_challenge_split(split, dataroot=nusc.dataroot)
        self.config = PredictionConfig.deserialize(
            open(f"nuscenes/eval/prediction/configs/{config_name}").read(), self.helper
        )

    def evaluate(self, predictions: dict[str, np.ndarray]) -> PredictionMetrics:
        """predictions: {token: (K, T, 2)}"""
        ades, fdes, misses = [], [], []
        for token in self.tokens:
            if token not in predictions:
                continue
            inst, sample = token.split("_")
            gt = self.helper.get_future_for_agent(inst, sample, seconds=6, in_agent_frame=True)
            pred = predictions[token]
            ades.append(compute_ade(pred, gt))
            fdes.append(compute_fde(pred, gt))
            misses.append(float(fdes[-1] > 2.0))
        return PredictionMetrics(
            ade_k1=float(np.mean(ades)),
            ade_k5=float(np.mean(ades)),
            fde_k1=float(np.mean(fdes)),
            fde_k5=float(np.mean(fdes)),
            miss_rate=float(np.mean(misses)),
        )
