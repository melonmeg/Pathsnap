"""Upload-first production inference pipeline for PathSnap."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import config as C
from .features import build_candidate_table, feature_matrix
from .hmm import fill_route, viterbi
from .io import Trip
from .network import Network


class ModelLoadError(RuntimeError):
    pass


class DependencyError(RuntimeError):
    pass


class ModelBundle:
    """Load the final XGBoost classifier and HMM configuration."""

    def __init__(self, artifact_dir: str | os.PathLike[str] | None = None):
        root = Path(__file__).resolve().parents[2]
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else root / "final_model"
        meta_path = self.artifact_dir / "model_meta.json"
        model_path = self.artifact_dir / "model.joblib"
        hmm_path = self.artifact_dir / "hmm_config.json"

        missing = [str(p) for p in (meta_path, model_path, hmm_path) if not p.is_file()]
        if missing:
            raise ModelLoadError("Missing production model artifact(s): " + ", ".join(missing))

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                self.meta = json.load(f)
            with hmm_path.open("r", encoding="utf-8") as f:
                self.hmm_cfg = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise ModelLoadError(f"Failed to read model configuration: {exc}") from exc

        features = self.meta.get("features")
        if features is None:
            raise ModelLoadError("model_meta.json is missing the required 'features' list")
        if features != C.FEATURES:
            raise ModelLoadError("Feature list in model_meta.json differs from config.FEATURES")
        self.features = features

        kind = self.meta.get("kind")
        if kind != "joblib":
            raise ModelLoadError(f"Unsupported model kind '{kind}'; expected 'joblib'")

        for dependency in ("joblib", "xgboost", "numpy", "pandas", "scipy", "sklearn"):
            try:
                __import__(dependency)
            except ImportError as exc:
                raise DependencyError(
                    f"Missing required dependency '{dependency}'. Install backend requirements."
                ) from exc
        try:
            import joblib
            self.model = joblib.load(model_path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {model_path}: {exc}") from exc

        if not hasattr(self.model, "predict_proba"):
            raise ModelLoadError("Loaded model does not provide predict_proba")
        self.radius = float(self.meta.get("radius", C.CAND_RADIUS_M))
        self.max_k = int(self.meta.get("max_k", C.MAX_CANDIDATES))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        try:
            probabilities = self.model.predict_proba(X)
        except Exception as exc:
            raise RuntimeError(f"XGBoost predict_proba failed: {exc}") from exc
        probabilities = np.asarray(probabilities)
        if probabilities.ndim != 2:
            raise RuntimeError("XGBoost predict_proba returned invalid output")
        if probabilities.shape[1] == 2:
            return probabilities[:, 1]
        if probabilities.shape[1] == 1:
            return probabilities[:, 0]
        raise RuntimeError("XGBoost predict_proba returned unexpected class columns")


def trip_from_parsed(parsed: Mapping[str, Any], trip_id: str = "upload") -> Trip:
    """Build the source-of-truth Trip type from parsed upload data."""
    track = np.asarray(parsed["timestamps"], dtype=np.float64)
    gps = np.asarray(parsed["gps_longitude_latitude"], dtype=np.float64)
    if gps.shape != (len(track), 2) or len(track) == 0:
        raise ValueError("Parsed uploaded trip has invalid track data")
    track_array = np.column_stack((gps[:, 0], gps[:, 1], track))
    nodes = np.asarray(parsed["nodes"], dtype=np.float64)
    arcs = np.asarray(parsed["arcs"], dtype=np.int64).reshape(-1, 2)
    # Ground-truth route IDs are retained by the API for display/evaluation only.
    # Inference receives no reference route.
    return Trip(trip_id, track_array, nodes, arcs, None)


def predict_parsed_trip(parsed: Mapping[str, Any], bundle: ModelBundle | None = None, use_hmm: bool = True) -> dict:
    """Run candidate generation, XGBoost, and optional HMM directly from parsed uploads."""
    try:
        import pandas  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as exc:
        raise DependencyError(f"Missing required inference dependency: {exc.name}") from exc

    if bundle is None:
        bundle = ModelBundle()
    trip = trip_from_parsed(parsed)
    t0 = time.time()
    net = Network(trip.nodes, trip.arcs, C.SAMPLE_STEP_M)
    df = build_candidate_table(trip, net, radius=bundle.radius, max_k=bundle.max_k,
                               min_k=C.MIN_CANDIDATES)
    if df.empty:
        raise RuntimeError("Candidate generation produced no candidates")
    prob = bundle.predict_proba(feature_matrix(df))

    obs = df["obs"].to_numpy(dtype=np.int64)
    arc = df["arc"].to_numpy(dtype=np.int64)
    n_obs = len(trip.track)
    # Highest probability candidate by observation.
    chosen = np.full(n_obs, -1, dtype=np.int64)
    for i in range(n_obs):
        rows = np.flatnonzero(obs == i)
        if rows.size:
            chosen[i] = rows[int(np.argmax(prob[rows]))]

    xy = net.project(trip.track[:, :2])
    if use_hmm:
        rows = viterbi(net, xy, trip.track[:, 2], obs, arc, df["along"].to_numpy(), prob, bundle.hmm_cfg)
    else:
        rows = chosen
    pred = np.where(rows >= 0, arc[np.maximum(rows, 0)], -1)
    # Reconstruct a connected route from model-selected arcs using only the
    # road network. This does not read or substitute the reference route.
    route = fill_route(net, pred)

    return {
        "route_arc_ids": [int(a) for a in route],
        "model": "xgboost_viterbi_v1",
        "is_demo": False,
        "net": net,
        "df": df,
        "prob": prob,
        "xy": xy,
        "pred": pred,
        "route": route,
        "timing": {"total": time.time() - t0},
    }
