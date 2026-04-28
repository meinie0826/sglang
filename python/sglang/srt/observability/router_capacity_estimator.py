from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import requests


logger = logging.getLogger(__name__)


class RouterCapacityEstimator:
    def __init__(
        self,
        decode_urls: List[str],
        decode_weights: Optional[List[float]] = None,
        request_timeout_seconds: float = 2.0,
        log_path: Optional[Path] = None,
    ):
        if not decode_urls:
            raise ValueError("at least one decode url is required")
        self.decode_urls = [url.rstrip("/") for url in decode_urls]
        if decode_weights is None:
            decode_weights = [1.0] * len(self.decode_urls)
        if len(decode_weights) != len(self.decode_urls):
            raise ValueError("decode_weights must match decode_urls")
        total_weight = sum(max(0.0, float(weight)) for weight in decode_weights)
        if total_weight <= 0.0:
            raise ValueError("decode_weights must contain a positive value")
        self.decode_weights = [
            max(0.0, float(weight)) / total_weight for weight in decode_weights
        ]
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.log_path = log_path

    @staticmethod
    def _append_jsonl(path: Optional[Path], record: dict) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            logger.exception("failed to write router capacity log to %s", path)

    @staticmethod
    def _weighted_median(values: List[float], weights: List[float]) -> float:
        pairs = sorted(zip(values, weights), key=lambda item: item[0])
        total = sum(weight for _, weight in pairs)
        threshold = total * 0.5
        acc = 0.0
        for value, weight in pairs:
            acc += weight
            if acc >= threshold:
                return float(value)
        return float(pairs[-1][0])

    def _request_decode_prediction(
        self,
        decode_url: str,
        decode_future_qps: float,
        horizon_seconds: Optional[float],
    ) -> dict:
        payload: Dict[str, float] = {"future_qps": float(decode_future_qps)}
        if horizon_seconds is not None:
            payload["horizon_seconds"] = float(horizon_seconds)
        response = requests.post(
            f"{decode_url}/predict_window_ttft",
            json=payload,
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def predict_router_ttft(
        self,
        future_qps: float,
        horizon_seconds: Optional[float] = None,
        aggregate: Literal["weighted_median", "max"] = "max",
    ) -> dict:
        decode_predictions = []
        errors = []
        for decode_url, weight in zip(self.decode_urls, self.decode_weights):
            decode_future_qps = max(0.0, float(future_qps)) * weight
            try:
                prediction = self._request_decode_prediction(
                    decode_url=decode_url,
                    decode_future_qps=decode_future_qps,
                    horizon_seconds=horizon_seconds,
                )
                predicted_ttft_ms = prediction.get("predicted_window_p50_ttft_ms")
                if predicted_ttft_ms is None:
                    raise RuntimeError(
                        f"decode predictor unavailable: {prediction.get('reason')}"
                    )
                decode_predictions.append(
                    {
                        "decode_url": decode_url,
                        "traffic_weight": float(weight),
                        "future_qps": float(decode_future_qps),
                        "predicted_window_p50_ttft_ms": float(predicted_ttft_ms),
                        "model_type": prediction.get("model_type"),
                        "model_version": prediction.get("window_ttft_model_version"),
                        "window_ttft_predictor_ready": bool(
                            prediction.get("window_ttft_predictor_ready", False)
                        ),
                        "current_window_predictor_ready": bool(
                            prediction.get("current_window_predictor_ready", False)
                        ),
                        "latest_window_p50_ttft_ms": prediction.get(
                            "latest_window_p50_ttft_ms"
                        ),
                        "recent_accuracy": prediction.get("recent_accuracy"),
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "decode_url": decode_url,
                        "traffic_weight": float(weight),
                        "future_qps": float(decode_future_qps),
                        "error": str(exc),
                    }
                )

        if not decode_predictions:
            raise RuntimeError(
                json.dumps(
                    {
                        "reason": "no_decode_predictions",
                        "decode_errors": errors,
                    }
                )
            )

        values = [
            item["predicted_window_p50_ttft_ms"] for item in decode_predictions
        ]
        weights = [item["traffic_weight"] for item in decode_predictions]
        if aggregate == "max":
            router_p50_ttft_ms = max(values)
        else:
            router_p50_ttft_ms = self._weighted_median(values, weights)

        model_ready = all(
            item["window_ttft_predictor_ready"] for item in decode_predictions
        )
        current_model_ready = all(
            item["current_window_predictor_ready"] for item in decode_predictions
        )
        response = {
            "router_window_ttft_predictor_ready": bool(model_ready),
            "router_current_window_predictor_ready": bool(current_model_ready),
            "future_qps": float(max(0.0, future_qps)),
            "horizon_seconds": horizon_seconds,
            "aggregate": aggregate,
            "predicted_router_p50_ttft_ms": float(router_p50_ttft_ms),
            "decode_predictions": decode_predictions,
            "decode_errors": errors,
        }
        self._append_jsonl(
            self.log_path,
            {
                "event": "router_ttft_prediction",
                "time": time.time(),
                **response,
            },
        )
        return response

    def estimate_capacity(
        self,
        slo_p50_ttft_ms: float,
        qps_min: float,
        qps_max: float,
        qps_step: float,
        horizon_seconds: Optional[float] = None,
        aggregate: Literal["weighted_median", "max"] = "max",
    ) -> dict:
        if qps_max < qps_min:
            raise ValueError("qps_max must be greater than or equal to qps_min")

        candidates = []
        qps = float(qps_min)
        # Add a small epsilon so a qps_max that lands on the grid is included.
        while qps <= float(qps_max) + 1e-9:
            prediction = self.predict_router_ttft(
                future_qps=qps,
                horizon_seconds=horizon_seconds,
                aggregate=aggregate,
            )
            candidates.append(prediction)
            qps += float(qps_step)

        feasible = [
            item
            for item in candidates
            if item["predicted_router_p50_ttft_ms"] <= float(slo_p50_ttft_ms)
        ]
        best = feasible[-1] if feasible else None
        first_violation = None
        if best is not None:
            for item in candidates:
                if (
                    item["future_qps"] > best["future_qps"]
                    and item["predicted_router_p50_ttft_ms"]
                    > float(slo_p50_ttft_ms)
                ):
                    first_violation = item
                    break
        else:
            for item in candidates:
                if item["predicted_router_p50_ttft_ms"] > float(slo_p50_ttft_ms):
                    first_violation = item
                    break

        response = {
            "slo_p50_ttft_ms": float(slo_p50_ttft_ms),
            "qps_min": float(qps_min),
            "qps_max": float(qps_max),
            "qps_step": float(qps_step),
            "horizon_seconds": horizon_seconds,
            "aggregate": aggregate,
            "estimated_max_qps": (
                float(best["future_qps"]) if best is not None else None
            ),
            "predicted_ttft_at_estimated_max_qps_ms": (
                float(best["predicted_router_p50_ttft_ms"])
                if best is not None
                else None
            ),
            "first_violation_qps": (
                float(first_violation["future_qps"])
                if first_violation is not None
                else None
            ),
            "predicted_ttft_at_first_violation_ms": (
                float(first_violation["predicted_router_p50_ttft_ms"])
                if first_violation is not None
                else None
            ),
            "candidate_count": len(candidates),
            "candidates": [
                {
                    "future_qps": item["future_qps"],
                    "predicted_router_p50_ttft_ms": item[
                        "predicted_router_p50_ttft_ms"
                    ],
                    "router_window_ttft_predictor_ready": item[
                        "router_window_ttft_predictor_ready"
                    ],
                }
                for item in candidates
            ],
            "best_prediction": best,
            "first_violation_prediction": first_violation,
        }
        self._append_jsonl(
            self.log_path,
            {
                "event": "router_ttft_capacity_estimate",
                "time": time.time(),
                **response,
            },
        )
        return response

    def status(self) -> dict:
        decode_statuses = []
        for decode_url, weight in zip(self.decode_urls, self.decode_weights):
            try:
                response = requests.get(
                    f"{decode_url}/window_ttft_predictor_status",
                    timeout=self.request_timeout_seconds,
                )
                response.raise_for_status()
                status = response.json()
                decode_statuses.append(
                    {
                        "decode_url": decode_url,
                        "traffic_weight": float(weight),
                        "ok": True,
                        "status": status,
                    }
                )
            except Exception as exc:
                decode_statuses.append(
                    {
                        "decode_url": decode_url,
                        "traffic_weight": float(weight),
                        "ok": False,
                        "error": str(exc),
                    }
                )
        return {
            "decode_urls": self.decode_urls,
            "decode_weights": self.decode_weights,
            "decode_statuses": decode_statuses,
        }


def create_app(estimator: RouterCapacityEstimator):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class RouterWindowTTFTPredictionReq(BaseModel):
        future_qps: float = Field(..., ge=0.0)
        horizon_seconds: Optional[float] = Field(default=None, ge=0.0)
        aggregate: Literal["weighted_median", "max"] = "max"

    class RouterTTFTCapacityReq(BaseModel):
        slo_p50_ttft_ms: float = Field(..., gt=0.0)
        qps_min: float = Field(default=0.0, ge=0.0)
        qps_max: float = Field(default=10.0, ge=0.0)
        qps_step: float = Field(default=0.1, gt=0.0)
        horizon_seconds: Optional[float] = Field(default=None, ge=0.0)
        aggregate: Literal["weighted_median", "max"] = "max"

    app = FastAPI()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/router_ttft_capacity_status")
    def router_ttft_capacity_status() -> dict:
        return estimator.status()

    @app.post("/predict_router_window_ttft")
    def predict_router_window_ttft(obj: RouterWindowTTFTPredictionReq) -> dict:
        try:
            return estimator.predict_router_ttft(
                future_qps=obj.future_qps,
                horizon_seconds=obj.horizon_seconds,
                aggregate=obj.aggregate,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/estimate_ttft_capacity")
    def estimate_ttft_capacity(obj: RouterTTFTCapacityReq) -> dict:
        try:
            return estimator.estimate_capacity(
                slo_p50_ttft_ms=obj.slo_p50_ttft_ms,
                qps_min=obj.qps_min,
                qps_max=obj.qps_max,
                qps_step=obj.qps_step,
                horizon_seconds=obj.horizon_seconds,
                aggregate=obj.aggregate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _parse_csv(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_weights(raw: Optional[str]) -> Optional[List[float]]:
    if not raw:
        return None
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Router-level TTFT capacity estimator for PD deployments."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30001)
    parser.add_argument(
        "--decode-urls",
        required=True,
        help="Comma-separated decode server base URLs.",
    )
    parser.add_argument(
        "--decode-weights",
        default=None,
        help="Optional comma-separated traffic weights for decode URLs.",
    )
    parser.add_argument("--request-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    estimator = RouterCapacityEstimator(
        decode_urls=_parse_csv(args.decode_urls),
        decode_weights=_parse_weights(args.decode_weights),
        request_timeout_seconds=args.request_timeout_seconds,
        log_path=Path(args.log_path) if args.log_path else None,
    )
    app = create_app(estimator)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
