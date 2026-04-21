from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("/tmp/sglang_window_ttft_model.json")
DEFAULT_EVENT_LOG_PATH = Path("/tmp/sglang_window_ttft_events.jsonl")


@dataclass
class CompletedRequestRecord:
    ts: float
    seqlen: int
    cachelen: int
    extend_tokens: int
    e2e_latency_ms: float


class OnlineRequestLatencyPredictor:
    """Window-level predictor for window p50 TTFT.

    Target:
        p50 of request e2e latency within the current aggregation window.
    """

    FEATURE_NAMES = [
        "active_requests",
        "arrival_qps_60s",
        "finished_qps_60s",
        "finished_count_60s",
        "window_mean_seqlen_60s",
        "window_p90_seqlen_60s",
        "window_mean_cachelen_60s",
        "window_p90_cachelen_60s",
        "window_mean_extend_tokens_60s",
        "window_p90_extend_tokens_60s",
        "window_mean_ttft_ms_60s",
        "window_p90_ttft_ms_60s",
    ]

    def __init__(
        self,
        model_path: Optional[Path] = None,
        event_log_path: Optional[Path] = None,
        aggregation_window_seconds: Optional[float] = None,
        training_window_seconds: Optional[float] = None,
        retrain_interval_seconds: Optional[float] = None,
        min_train_samples: int = 20,
        min_window_requests: int = 5,
        ridge_lambda: float = 1e-3,
    ):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.event_log_path = event_log_path or DEFAULT_EVENT_LOG_PATH
        self.aggregation_window_seconds = (
            aggregation_window_seconds
            if aggregation_window_seconds is not None
            else envs.SGLANG_WINDOW_TTFT_PREDICTOR_WINDOW_SECONDS.get()
        )
        self.training_window_seconds = (
            training_window_seconds
            if training_window_seconds is not None
            else envs.SGLANG_WINDOW_TTFT_PREDICTOR_TRAINING_WINDOW_SECONDS.get()
        )
        self.retrain_interval_seconds = (
            retrain_interval_seconds
            if retrain_interval_seconds is not None
            else envs.SGLANG_WINDOW_TTFT_PREDICTOR_RETRAIN_INTERVAL_SECONDS.get()
        )
        self.min_train_samples = min_train_samples
        self.min_window_requests = min_window_requests
        self.ridge_lambda = ridge_lambda

        self._arrival_times: Deque[float] = deque()
        self._completed_records: Deque[CompletedRequestRecord] = deque()
        self._training_samples: Deque[dict] = deque()

        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None
        self._model_version: int = 0
        self._last_retrain_time: float = 0.0
        self._last_retrain_sample_count: int = 0
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._retrain_thread = threading.Thread(
            target=self._retrain_loop,
            name="sglang-window-ttft-retrain",
            daemon=True,
        )

        self._load_model()
        self._retrain_thread.start()

    def _retrain_loop(self) -> None:
        while not self._stop_event.wait(self.retrain_interval_seconds):
            try:
                self.maybe_retrain()
            except Exception:
                logger.exception("[WindowTTFTPredictor] periodic retrain failed")

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with open(self.model_path, encoding="utf-8") as f:
                data = json.load(f)
            self._model_version = int(data.get("model_version", 0))
            self._mean = np.array(data["feature_mean"], dtype=np.float64)
            self._std = np.array(data["feature_std"], dtype=np.float64)
            self._weights = np.array(data["weights"], dtype=np.float64)
            self._last_retrain_time = float(data.get("last_retrain_time", 0.0))
        except Exception:
            logger.exception(
                "[WindowTTFTPredictor] failed to load model from %s",
                self.model_path,
            )

    def _persist_model(self, sample_count: int) -> None:
        if self._weights is None or self._mean is None or self._std is None:
            return
        payload = {
            "model_version": self._model_version,
            "feature_names": self.FEATURE_NAMES,
            "feature_mean": self._mean.tolist(),
            "feature_std": self._std.tolist(),
            "weights": self._weights.tolist(),
            "sample_count": sample_count,
            "last_retrain_time": self._last_retrain_time,
            "aggregation_window_seconds": self.aggregation_window_seconds,
            "training_window_seconds": self.training_window_seconds,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _log_event(self, record: dict) -> None:
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception(
                "[WindowTTFTPredictor] failed to write event log to %s",
                self.event_log_path,
            )

    def _prune_locked(self, now: float) -> None:
        arrival_cutoff = now - self.aggregation_window_seconds
        while self._arrival_times and self._arrival_times[0] < arrival_cutoff:
            self._arrival_times.popleft()

        completed_cutoff = now - self.aggregation_window_seconds
        while self._completed_records and self._completed_records[0].ts < completed_cutoff:
            self._completed_records.popleft()

        training_cutoff = now - self.training_window_seconds
        while self._training_samples and self._training_samples[0]["ts"] < training_cutoff:
            self._training_samples.popleft()

    @staticmethod
    def _percentile(values: np.ndarray, q: float) -> float:
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, q))

    def _build_window_summary_locked(self, active_requests: int) -> Optional[dict]:
        finished_count = len(self._completed_records)
        if finished_count < self.min_window_requests:
            return None

        seqlens = np.array([r.seqlen for r in self._completed_records], dtype=np.float64)
        cachelens = np.array([r.cachelen for r in self._completed_records], dtype=np.float64)
        extend_tokens = np.array(
            [r.extend_tokens for r in self._completed_records], dtype=np.float64
        )
        ttfts = np.array(
            [r.e2e_latency_ms for r in self._completed_records], dtype=np.float64
        )

        features = {
            "active_requests": float(max(active_requests, 0)),
            "arrival_qps_60s": len(self._arrival_times) / self.aggregation_window_seconds,
            "finished_qps_60s": finished_count / self.aggregation_window_seconds,
            "finished_count_60s": float(finished_count),
            "window_mean_seqlen_60s": float(seqlens.mean()),
            "window_p90_seqlen_60s": self._percentile(seqlens, 90),
            "window_mean_cachelen_60s": float(cachelens.mean()),
            "window_p90_cachelen_60s": self._percentile(cachelens, 90),
            "window_mean_extend_tokens_60s": float(extend_tokens.mean()),
            "window_p90_extend_tokens_60s": self._percentile(extend_tokens, 90),
            "window_mean_ttft_ms_60s": float(ttfts.mean()),
            "window_p90_ttft_ms_60s": self._percentile(ttfts, 90),
        }
        observed_p50_ttft_ms = self._percentile(ttfts, 50)
        return {
            "features": features,
            "observed_p50_ttft_ms": observed_p50_ttft_ms,
        }

    def _feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        return np.array(
            [float(features.get(name, 0.0)) for name in self.FEATURE_NAMES],
            dtype=np.float64,
        )

    def _predict_locked(self, features: Dict[str, float]) -> float:
        if self._weights is None or self._mean is None or self._std is None:
            # Warm start heuristic: use current window mean TTFT before model is trained.
            return max(0.0, float(features.get("window_mean_ttft_ms_60s", 0.0)))
        x = self._feature_vector(features)
        x_norm = (x - self._mean) / self._std
        x_aug = np.concatenate([np.ones(1, dtype=np.float64), x_norm])
        return max(0.0, float(np.dot(x_aug, self._weights)))

    def observe_request_arrival(self, now: Optional[float] = None) -> None:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._prune_locked(now)
            self._arrival_times.append(now)

    def observe_request_completion(
        self,
        seqlen: int,
        cachelen: int,
        actual_latency_ms: float,
        active_requests: int,
        finish_ts: Optional[float] = None,
    ) -> Dict[str, float]:
        finish_ts = time.perf_counter() if finish_ts is None else finish_ts
        seqlen = max(0, int(seqlen))
        cachelen = max(0, min(int(cachelen), seqlen))
        actual_latency_ms = max(0.0, float(actual_latency_ms))
        extend_tokens = max(0, seqlen - cachelen)

        with self._lock:
            self._prune_locked(finish_ts)
            self._completed_records.append(
                CompletedRequestRecord(
                    ts=finish_ts,
                    seqlen=seqlen,
                    cachelen=cachelen,
                    extend_tokens=extend_tokens,
                    e2e_latency_ms=actual_latency_ms,
                )
            )
            summary = self._build_window_summary_locked(active_requests)
            if summary is None:
                return {}
            predicted_window_p50_ttft_ms = self._predict_locked(summary["features"])
            self._training_samples.append(
                {
                    "ts": finish_ts,
                    "features": dict(summary["features"]),
                    "label": float(summary["observed_p50_ttft_ms"]),
                }
            )
            model_ready = self._weights is not None
            model_version = self._model_version

        snapshot = {
            "window_ttft_window_seconds": float(self.aggregation_window_seconds),
            "window_p50_ttft_ms": float(summary["observed_p50_ttft_ms"]),
            "predicted_window_p50_ttft_ms": float(predicted_window_p50_ttft_ms),
            "window_ttft_abs_error_ms": abs(
                float(summary["observed_p50_ttft_ms"])
                - float(predicted_window_p50_ttft_ms)
            ),
            "window_ttft_predictor_ready": bool(model_ready),
            "window_ttft_model_version": int(model_version),
            "arrival_qps_60s": float(summary["features"]["arrival_qps_60s"]),
            "finished_qps_60s": float(summary["features"]["finished_qps_60s"]),
            "window_finished_count_60s": float(summary["features"]["finished_count_60s"]),
            "window_mean_seqlen_60s": float(summary["features"]["window_mean_seqlen_60s"]),
            "window_mean_cachelen_60s": float(summary["features"]["window_mean_cachelen_60s"]),
            "window_mean_ttft_ms_60s": float(summary["features"]["window_mean_ttft_ms_60s"]),
            "window_p90_ttft_ms_60s": float(summary["features"]["window_p90_ttft_ms_60s"]),
        }
        self._log_event(
            {
                "event": "window_snapshot",
                "time": time.time(),
                **snapshot,
            }
        )
        self.maybe_retrain()
        return snapshot

    def maybe_retrain(self) -> None:
        now = time.perf_counter()
        with self._lock:
            self._prune_locked(now)
            sample_count = len(self._training_samples)
            if sample_count < self.min_train_samples:
                return
            if sample_count == self._last_retrain_sample_count:
                return
            if now - self._last_retrain_time < self.retrain_interval_seconds:
                return

            x = np.array(
                [
                    self._feature_vector(sample["features"])
                    for sample in self._training_samples
                ],
                dtype=np.float64,
            )
            y = np.array(
                [float(sample["label"]) for sample in self._training_samples],
                dtype=np.float64,
            )
            mean = x.mean(axis=0)
            std = x.std(axis=0)
            std[std < 1e-6] = 1.0
            x_norm = (x - mean) / std
            x_aug = np.concatenate(
                [np.ones((x_norm.shape[0], 1), dtype=np.float64), x_norm],
                axis=1,
            )
            reg = np.eye(x_aug.shape[1], dtype=np.float64) * self.ridge_lambda
            reg[0, 0] = 0.0
            try:
                weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
            except np.linalg.LinAlgError:
                logger.warning(
                    "[WindowTTFTPredictor] retrain skipped due to singular matrix"
                )
                return

            preds = x_aug @ weights
            mae_ms = float(np.mean(np.abs(preds - y)))
            self._mean = mean
            self._std = std
            self._weights = weights
            self._model_version += 1
            self._last_retrain_time = now
            self._last_retrain_sample_count = sample_count
            self._persist_model(sample_count)

        self._log_event(
            {
                "event": "retrain",
                "time": time.time(),
                "model_version": self._model_version,
                "sample_count": sample_count,
                "mae_ms": mae_ms,
            }
        )


_global_request_latency_predictor: Optional[OnlineRequestLatencyPredictor] = None


def get_request_latency_predictor() -> OnlineRequestLatencyPredictor:
    global _global_request_latency_predictor
    if _global_request_latency_predictor is None:
        _global_request_latency_predictor = OnlineRequestLatencyPredictor()
    return _global_request_latency_predictor
