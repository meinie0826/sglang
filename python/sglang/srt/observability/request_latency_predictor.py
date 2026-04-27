from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


DEFAULT_MODEL_PATH = Path("/tmp/sglang_window_ttft_model.json")
DEFAULT_EVENT_LOG_PATH = Path("/tmp/sglang_window_ttft_events.jsonl")
DEFAULT_FUTURE_QPS_LOG_PATH = Path("/tmp/sglang_window_ttft_future_qps.jsonl")


@dataclass
class CompletedRequestRecord:
    ts: float
    seqlen: int
    cachelen: int
    extend_tokens: int
    ttft_ms: float


@dataclass
class WindowPredictionRecord:
    ts: float
    actual_p50_ttft_ms: float
    predicted_p50_ttft_ms: float


@dataclass
class FutureQpsPredictionRecord:
    prediction_ts: float
    target_ts: float
    future_qps: float
    predicted_p50_ttft_ms: float
    model_version: int
    base_window_p50_ttft_ms: float
    base_features: Dict[str, float]
    scenario_features: Dict[str, float]


class OnlineRequestLatencyPredictor:
    """Window-level predictor for current-window p50 TTFT.

    Target:
        p50 of request TTFT within the current aggregation window.

    Inference:
        Uses the latest window request features, with caller-provided qps replacing
        the qps features, to estimate the current-window p50 TTFT under that qps.
    """

    FEATURE_NAMES = [
        "active_requests",
        "arrival_qps_60s",
        "finished_qps_60s",
        "backlog_growth_qps_60s",
        "finished_count_60s",
        "window_mean_seqlen_60s",
        "window_p90_seqlen_60s",
        "window_mean_cachelen_60s",
        "window_p90_cachelen_60s",
        "window_mean_extend_tokens_60s",
        "window_p90_extend_tokens_60s",
        "window_mean_ttft_ms_60s",
        "window_p90_ttft_ms_60s",
        "delta_active_requests",
        "delta_arrival_qps_60s",
        "delta_finished_qps_60s",
        "delta_backlog_growth_qps_60s",
        "delta_finished_count_60s",
        "delta_window_mean_ttft_ms_60s",
        "delta_window_p90_ttft_ms_60s",
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
        xgb_n_estimators: int = 96,
        xgb_max_depth: int = 4,
        xgb_learning_rate: float = 0.08,
    ):
        configured_model_path = envs.SGLANG_WINDOW_TTFT_PREDICTOR_MODEL_PATH.get()
        configured_event_log_path = (
            envs.SGLANG_WINDOW_TTFT_PREDICTOR_EVENT_LOG_PATH.get()
        )
        if model_path is not None:
            self.model_path = model_path
        elif configured_model_path:
            self.model_path = Path(configured_model_path)
        else:
            self.model_path = DEFAULT_MODEL_PATH

        if event_log_path is not None:
            self.event_log_path = event_log_path
        elif configured_event_log_path:
            self.event_log_path = Path(configured_event_log_path)
        else:
            self.event_log_path = DEFAULT_EVENT_LOG_PATH

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
        configured_future_qps_log_path = (
            envs.SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_LOG_PATH.get()
        )
        if configured_future_qps_log_path:
            self.future_qps_log_path = Path(configured_future_qps_log_path)
        elif self.event_log_path:
            self.future_qps_log_path = self.event_log_path.with_name(
                "sglang_window_ttft_future_qps.jsonl"
            )
        else:
            self.future_qps_log_path = DEFAULT_FUTURE_QPS_LOG_PATH
        self.future_qps_values = self._parse_future_qps_values(
            envs.SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_VALUES.get()
        )
        self.future_qps_interval_seconds = (
            envs.SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_INTERVAL_SECONDS.get()
        )
        self.future_qps_horizon_seconds = max(
            0.0,
            float(
                envs.SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_HORIZON_SECONDS.get()
            ),
        )
        self.future_qps_match_tolerance = max(
            0.0,
            float(
                envs.SGLANG_WINDOW_TTFT_PREDICTOR_FUTURE_QPS_MATCH_TOLERANCE.get()
            ),
        )
        self.min_train_samples = min_train_samples
        self.min_window_requests = min_window_requests
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate

        self._arrival_times: Deque[float] = deque()
        self._completed_records: Deque[CompletedRequestRecord] = deque()
        self._training_samples: Deque[dict] = deque()
        self._prediction_history: Deque[WindowPredictionRecord] = deque()
        self._pending_future_predictions: Deque[FutureQpsPredictionRecord] = deque()

        self._model: Optional[XGBRegressor] = None
        self._model_version: int = 0
        self._last_retrain_time: float = 0.0
        self._last_retrain_sample_count: int = 0
        self._last_future_prediction_emit_time: float = 0.0
        self._latest_window_summary: Optional[dict] = None
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._retrain_thread = threading.Thread(
            target=self._retrain_loop,
            name="sglang-window-ttft-retrain",
            daemon=True,
        )

        self._load_model()
        self._retrain_thread.start()

    def _make_regressor(self) -> XGBRegressor:
        if XGBRegressor is None:
            raise RuntimeError(
                "xgboost is not installed. Please install the python package dependency "
                "for the window TTFT predictor."
            )
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.xgb_n_estimators,
            max_depth=self.xgb_max_depth,
            learning_rate=self.xgb_learning_rate,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=1.0,
            random_state=0,
            n_jobs=1,
        )

    @staticmethod
    def _parse_future_qps_values(raw: str) -> Tuple[float, ...]:
        if not raw:
            return tuple()
        values = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(max(0.0, float(part)))
            except ValueError:
                logger.warning(
                    "[WindowTTFTPredictor] invalid future qps value %r, skipping",
                    part,
                )
        return tuple(values)

    def _retrain_loop(self) -> None:
        while not self._stop_event.wait(self.retrain_interval_seconds):
            try:
                self.maybe_retrain()
            except Exception:
                logger.exception("[WindowTTFTPredictor] periodic retrain failed")

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        if XGBRegressor is None:
            logger.warning(
                "[WindowTTFTPredictor] xgboost unavailable, skipping model load from %s",
                self.model_path,
            )
            return
        try:
            model = self._make_regressor()
            model.load_model(str(self.model_path))
            booster = model.get_booster()
            attrs = booster.attributes()
            feature_names = json.loads(attrs.get("feature_names", "[]"))
            if feature_names and feature_names != self.FEATURE_NAMES:
                logger.warning(
                    "[WindowTTFTPredictor] feature name mismatch while loading %s; "
                    "expected=%s loaded=%s",
                    self.model_path,
                    self.FEATURE_NAMES,
                    feature_names,
                )
                return
            self._model = model
            self._model_version = int(attrs.get("model_version", "0"))
            self._last_retrain_time = float(attrs.get("last_retrain_time", "0.0"))
        except Exception:
            logger.exception(
                "[WindowTTFTPredictor] failed to load XGBoost model from %s",
                self.model_path,
            )

    def _persist_model(self, sample_count: int) -> None:
        if self._model is None:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        booster = self._model.get_booster()
        booster.set_attr(
            model_type="xgboost",
            model_version=str(self._model_version),
            last_retrain_time=str(self._last_retrain_time),
            sample_count=str(sample_count),
            aggregation_window_seconds=str(self.aggregation_window_seconds),
            training_window_seconds=str(self.training_window_seconds),
            feature_names=json.dumps(self.FEATURE_NAMES),
        )
        self._model.save_model(str(self.model_path))

    @staticmethod
    def _append_jsonl(path: Path, record: dict) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("[WindowTTFTPredictor] failed to write jsonl to %s", path)

    def _log_event(self, record: dict) -> None:
        self._append_jsonl(self.event_log_path, record)

    def _log_future_qps_event(self, record: dict) -> None:
        self._append_jsonl(self.future_qps_log_path, record)

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
        while self._prediction_history and self._prediction_history[0].ts < training_cutoff:
            self._prediction_history.popleft()
        while (
            self._pending_future_predictions
            and self._pending_future_predictions[0].target_ts < training_cutoff
        ):
            self._pending_future_predictions.popleft()

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
            [r.ttft_ms for r in self._completed_records], dtype=np.float64
        )

        features = {
            "active_requests": float(max(active_requests, 0)),
            "arrival_qps_60s": len(self._arrival_times) / self.aggregation_window_seconds,
            "finished_qps_60s": finished_count / self.aggregation_window_seconds,
            "backlog_growth_qps_60s": 0.0,
            "finished_count_60s": float(finished_count),
            "window_mean_seqlen_60s": float(seqlens.mean()),
            "window_p90_seqlen_60s": self._percentile(seqlens, 90),
            "window_mean_cachelen_60s": float(cachelens.mean()),
            "window_p90_cachelen_60s": self._percentile(cachelens, 90),
            "window_mean_extend_tokens_60s": float(extend_tokens.mean()),
            "window_p90_extend_tokens_60s": self._percentile(extend_tokens, 90),
            "window_mean_ttft_ms_60s": float(ttfts.mean()),
            "window_p90_ttft_ms_60s": self._percentile(ttfts, 90),
            "delta_active_requests": 0.0,
            "delta_arrival_qps_60s": 0.0,
            "delta_finished_qps_60s": 0.0,
            "delta_backlog_growth_qps_60s": 0.0,
            "delta_finished_count_60s": 0.0,
            "delta_window_mean_ttft_ms_60s": 0.0,
            "delta_window_p90_ttft_ms_60s": 0.0,
        }
        previous_features = None
        if self._latest_window_summary is not None:
            previous_features = self._latest_window_summary["features"]
        features = self._apply_trend_features(
            features=features,
            reference_features=previous_features,
        )
        observed_p50_ttft_ms = self._percentile(ttfts, 50)
        return {
            "features": features,
            "observed_p50_ttft_ms": observed_p50_ttft_ms,
        }

    def _apply_trend_features(
        self,
        features: Dict[str, float],
        reference_features: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        enriched = dict(features)
        enriched["backlog_growth_qps_60s"] = float(
            enriched.get("arrival_qps_60s", 0.0)
            - enriched.get("finished_qps_60s", 0.0)
        )
        if reference_features is None:
            return enriched

        enriched["delta_active_requests"] = float(
            enriched.get("active_requests", 0.0)
            - reference_features.get("active_requests", 0.0)
        )
        enriched["delta_arrival_qps_60s"] = float(
            enriched.get("arrival_qps_60s", 0.0)
            - reference_features.get("arrival_qps_60s", 0.0)
        )
        enriched["delta_finished_qps_60s"] = float(
            enriched.get("finished_qps_60s", 0.0)
            - reference_features.get("finished_qps_60s", 0.0)
        )
        enriched["delta_backlog_growth_qps_60s"] = float(
            enriched.get("backlog_growth_qps_60s", 0.0)
            - reference_features.get("backlog_growth_qps_60s", 0.0)
        )
        enriched["delta_finished_count_60s"] = float(
            enriched.get("finished_count_60s", 0.0)
            - reference_features.get("finished_count_60s", 0.0)
        )
        enriched["delta_window_mean_ttft_ms_60s"] = float(
            enriched.get("window_mean_ttft_ms_60s", 0.0)
            - reference_features.get("window_mean_ttft_ms_60s", 0.0)
        )
        enriched["delta_window_p90_ttft_ms_60s"] = float(
            enriched.get("window_p90_ttft_ms_60s", 0.0)
            - reference_features.get("window_p90_ttft_ms_60s", 0.0)
        )
        return enriched

    def _feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        return np.array(
            [float(features.get(name, 0.0)) for name in self.FEATURE_NAMES],
            dtype=np.float32,
        )

    def _predict_locked(self, features: Dict[str, float]) -> float:
        if self._model is None:
            return max(0.0, float(features.get("window_mean_ttft_ms_60s", 0.0)))
        x = self._feature_vector(features).reshape(1, -1)
        return max(0.0, float(self._model.predict(x)[0]))

    def _build_future_qps_features_locked(
        self,
        base_features: Dict[str, float],
        future_qps: float,
        feature_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        scenario = dict(base_features)
        scenario["arrival_qps_60s"] = float(max(future_qps, 0.0))
        scenario["finished_qps_60s"] = float(max(future_qps, 0.0))
        if feature_overrides:
            for key, value in feature_overrides.items():
                if key in self.FEATURE_NAMES:
                    scenario[key] = float(value)
        return self._apply_trend_features(
            features=scenario,
            reference_features=base_features,
        )

    def _accuracy_summary_locked(self) -> dict:
        if not self._prediction_history:
            return {
                "count": 0,
                "mae_ms": None,
                "mape_pct": None,
                "p50_abs_err_ms": None,
                "p90_abs_err_ms": None,
                "within_10pct_rate": None,
                "within_20pct_rate": None,
            }

        actual = np.array(
            [r.actual_p50_ttft_ms for r in self._prediction_history], dtype=np.float64
        )
        predicted = np.array(
            [r.predicted_p50_ttft_ms for r in self._prediction_history], dtype=np.float64
        )
        abs_err = np.abs(actual - predicted)
        rel_err = abs_err / np.maximum(actual, 1e-6)
        return {
            "count": int(len(self._prediction_history)),
            "mae_ms": float(abs_err.mean()),
            "mape_pct": float(rel_err.mean() * 100.0),
            "p50_abs_err_ms": self._percentile(abs_err, 50),
            "p90_abs_err_ms": self._percentile(abs_err, 90),
            "within_10pct_rate": float((rel_err <= 0.10).mean()),
            "within_20pct_rate": float((rel_err <= 0.20).mean()),
        }

    def _maybe_emit_future_qps_predictions_locked(
        self, now: float
    ) -> list[FutureQpsPredictionRecord]:
        if not self.future_qps_values:
            return []
        if self._latest_window_summary is None:
            return []
        if (
            self._last_future_prediction_emit_time > 0.0
            and now - self._last_future_prediction_emit_time
            < self.future_qps_interval_seconds
        ):
            return []

        base_features = dict(self._latest_window_summary["features"])
        base_window_p50_ttft_ms = float(
            self._latest_window_summary["observed_p50_ttft_ms"]
        )
        target_ts = now + self.future_qps_horizon_seconds
        records = []
        for future_qps in self.future_qps_values:
            scenario_features = self._build_future_qps_features_locked(
                base_features=base_features,
                future_qps=future_qps,
            )
            predicted_ms = self._predict_locked(scenario_features)
            record = FutureQpsPredictionRecord(
                prediction_ts=now,
                target_ts=target_ts,
                future_qps=float(future_qps),
                predicted_p50_ttft_ms=float(predicted_ms),
                model_version=int(self._model_version),
                base_window_p50_ttft_ms=base_window_p50_ttft_ms,
                base_features=base_features,
                scenario_features=scenario_features,
            )
            self._pending_future_predictions.append(record)
            records.append(record)

        self._last_future_prediction_emit_time = now
        return records

    def _resolve_future_qps_predictions_locked(
        self, now: float, actual_summary: dict
    ) -> list[dict]:
        resolved = []
        while self._pending_future_predictions and (
            self._pending_future_predictions[0].target_ts <= now
        ):
            record = self._pending_future_predictions.popleft()
            actual_qps = float(actual_summary["features"]["arrival_qps_60s"])
            actual_p50_ttft_ms = float(actual_summary["observed_p50_ttft_ms"])
            qps_abs_error = abs(actual_qps - record.future_qps)
            qps_match = qps_abs_error <= self.future_qps_match_tolerance
            raw_window_ttft_abs_error_ms = abs(
                actual_p50_ttft_ms - record.predicted_p50_ttft_ms
            )
            resolved.append(
                {
                    "event": "future_qps_evaluation",
                    "time": time.time(),
                    "prediction_monotonic_ts": record.prediction_ts,
                    "target_monotonic_ts": record.target_ts,
                    "future_qps": float(record.future_qps),
                    "future_qps_horizon_seconds": float(
                        self.future_qps_horizon_seconds
                    ),
                    "actual_arrival_qps_60s": actual_qps,
                    "actual_finished_qps_60s": float(
                        actual_summary["features"]["finished_qps_60s"]
                    ),
                    "predicted_window_p50_ttft_ms": float(
                        record.predicted_p50_ttft_ms
                    ),
                    "actual_window_p50_ttft_ms": actual_p50_ttft_ms,
                    "window_ttft_abs_error_ms": (
                        raw_window_ttft_abs_error_ms if qps_match else None
                    ),
                    "raw_window_ttft_abs_error_ms": raw_window_ttft_abs_error_ms,
                    "qps_abs_error": qps_abs_error,
                    "qps_match": qps_match,
                    "qps_match_tolerance": float(self.future_qps_match_tolerance),
                    "model_version": int(record.model_version),
                }
            )
        return resolved

    def observe_request_arrival(self, now: Optional[float] = None) -> None:
        now = time.perf_counter() if now is None else now
        with self._lock:
            self._prune_locked(now)
            self._arrival_times.append(now)

    def observe_request_completion(
        self,
        seqlen: int,
        cachelen: int,
        actual_ttft_ms: float,
        active_requests: int,
        finish_ts: Optional[float] = None,
    ) -> Dict[str, float]:
        finish_ts = time.perf_counter() if finish_ts is None else finish_ts
        seqlen = max(0, int(seqlen))
        cachelen = max(0, min(int(cachelen), seqlen))
        actual_ttft_ms = max(0.0, float(actual_ttft_ms))
        extend_tokens = max(0, seqlen - cachelen)

        with self._lock:
            self._prune_locked(finish_ts)
            self._completed_records.append(
                CompletedRequestRecord(
                    ts=finish_ts,
                    seqlen=seqlen,
                    cachelen=cachelen,
                    extend_tokens=extend_tokens,
                    ttft_ms=actual_ttft_ms,
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
            self._prediction_history.append(
                WindowPredictionRecord(
                    ts=finish_ts,
                    actual_p50_ttft_ms=float(summary["observed_p50_ttft_ms"]),
                    predicted_p50_ttft_ms=float(predicted_window_p50_ttft_ms),
                )
            )
            self._latest_window_summary = {
                "ts": finish_ts,
                "features": dict(summary["features"]),
                "observed_p50_ttft_ms": float(summary["observed_p50_ttft_ms"]),
            }
            future_qps_evaluations = self._resolve_future_qps_predictions_locked(
                finish_ts, summary
            )
            model_ready = self._model is not None
            model_version = self._model_version
            accuracy_summary = self._accuracy_summary_locked()
            future_qps_predictions = self._maybe_emit_future_qps_predictions_locked(
                finish_ts
            )

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
            "backlog_growth_qps_60s": float(
                summary["features"]["backlog_growth_qps_60s"]
            ),
            "window_finished_count_60s": float(summary["features"]["finished_count_60s"]),
            "window_mean_seqlen_60s": float(summary["features"]["window_mean_seqlen_60s"]),
            "window_mean_cachelen_60s": float(summary["features"]["window_mean_cachelen_60s"]),
            "window_mean_ttft_ms_60s": float(summary["features"]["window_mean_ttft_ms_60s"]),
            "window_p90_ttft_ms_60s": float(summary["features"]["window_p90_ttft_ms_60s"]),
            "delta_active_requests": float(summary["features"]["delta_active_requests"]),
            "delta_arrival_qps_60s": float(summary["features"]["delta_arrival_qps_60s"]),
            "delta_finished_qps_60s": float(
                summary["features"]["delta_finished_qps_60s"]
            ),
            "delta_window_mean_ttft_ms_60s": float(
                summary["features"]["delta_window_mean_ttft_ms_60s"]
            ),
            "delta_window_p90_ttft_ms_60s": float(
                summary["features"]["delta_window_p90_ttft_ms_60s"]
            ),
            "window_ttft_recent_eval_count": int(accuracy_summary["count"]),
            "window_ttft_recent_mae_ms": accuracy_summary["mae_ms"],
            "window_ttft_recent_mape_pct": accuracy_summary["mape_pct"],
        }
        self._log_event(
            {
                "event": "window_snapshot",
                "time": time.time(),
                **snapshot,
            }
        )
        for record in future_qps_predictions:
            self._log_future_qps_event(
                {
                    "event": "future_qps_prediction",
                    "time": time.time(),
                    "prediction_monotonic_ts": record.prediction_ts,
                    "target_monotonic_ts": record.target_ts,
                    "future_qps": float(record.future_qps),
                    "future_qps_horizon_seconds": float(
                        self.future_qps_horizon_seconds
                    ),
                    "predicted_window_p50_ttft_ms": float(
                        record.predicted_p50_ttft_ms
                    ),
                    "base_window_p50_ttft_ms": float(record.base_window_p50_ttft_ms),
                    "model_version": int(record.model_version),
                    "base_features": record.base_features,
                    "scenario_features": record.scenario_features,
                }
            )
        for record in future_qps_evaluations:
            self._log_future_qps_event(record)
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
                dtype=np.float32,
            )
            y = np.array(
                [float(sample["label"]) for sample in self._training_samples],
                dtype=np.float32,
            )
            model = self._make_regressor()
            model.fit(x, y, verbose=False)
            preds = model.predict(x)
            mae_ms = float(np.mean(np.abs(preds - y)))
            mape_pct = float(
                np.mean(np.abs(preds - y) / np.maximum(y, 1e-6)) * 100.0
            )
            self._model = model
            self._model_version += 1
            self._last_retrain_time = now
            self._last_retrain_sample_count = sample_count
            self._persist_model(sample_count)

        self._log_event(
            {
                "event": "retrain",
                "time": time.time(),
                "model_type": "xgboost",
                "model_version": self._model_version,
                "sample_count": sample_count,
                "mae_ms": mae_ms,
                "mape_pct": mape_pct,
            }
        )

    def predict_with_future_qps(
        self,
        future_qps: float,
        feature_overrides: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        with self._lock:
            if self._latest_window_summary is None:
                return {
                    "window_ttft_predictor_ready": False,
                    "reason": "no_window_summary",
                }

            scenario_features = self._build_future_qps_features_locked(
                base_features=self._latest_window_summary["features"],
                future_qps=future_qps,
                feature_overrides=feature_overrides,
            )
            predicted_ms = self._predict_locked(scenario_features)
            accuracy_summary = self._accuracy_summary_locked()
            latest_actual = float(self._latest_window_summary["observed_p50_ttft_ms"])
            model_ready = self._model is not None
            model_version = self._model_version

        return {
            "window_ttft_predictor_ready": bool(model_ready),
            "window_ttft_model_version": int(model_version),
            "future_qps": float(max(future_qps, 0.0)),
            "latest_window_p50_ttft_ms": latest_actual,
            "predicted_window_p50_ttft_ms": float(predicted_ms),
            "latest_window_features": scenario_features,
            "recent_accuracy": accuracy_summary,
        }

    def get_predictor_status(self) -> Dict[str, object]:
        with self._lock:
            accuracy_summary = self._accuracy_summary_locked()
            latest_window_summary = None
            if self._latest_window_summary is not None:
                latest_window_summary = {
                    "window_p50_ttft_ms": float(
                        self._latest_window_summary["observed_p50_ttft_ms"]
                    ),
                    "features": dict(self._latest_window_summary["features"]),
                }
            return {
                "window_ttft_predictor_ready": bool(self._model is not None),
                "window_ttft_model_version": int(self._model_version),
                "aggregation_window_seconds": float(self.aggregation_window_seconds),
                "training_window_seconds": float(self.training_window_seconds),
                "future_qps_values": list(self.future_qps_values),
                "future_qps_interval_seconds": float(
                    self.future_qps_interval_seconds
                ),
                "future_qps_horizon_seconds": float(
                    self.future_qps_horizon_seconds
                ),
                "future_qps_match_tolerance": float(
                    self.future_qps_match_tolerance
                ),
                "latest_window_summary": latest_window_summary,
                "recent_accuracy": accuracy_summary,
            }


_global_request_latency_predictor: Optional[OnlineRequestLatencyPredictor] = None


def get_request_latency_predictor() -> OnlineRequestLatencyPredictor:
    global _global_request_latency_predictor
    if _global_request_latency_predictor is None:
        _global_request_latency_predictor = OnlineRequestLatencyPredictor()
    return _global_request_latency_predictor
