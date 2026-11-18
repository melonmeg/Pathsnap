"""PathSnap API for backend test-dataset IDs and complete trip uploads."""

from __future__ import annotations

import sys
from pathlib import Path
import re

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pathsnap.model_stub import predict_route  # noqa: E402
from src.pathsnap.pipeline import DependencyError, ModelLoadError  # noqa: E402
from src.pathsnap.trip import TripValidationError, parse_trip, route_segments_for_ids  # noqa: E402

app = FastAPI(title="PathSnap", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEST_DATASET_DIR = ROOT / "dataset" / "evaluator_test"
TRIP_FILE_EXTENSIONS = ("track", "nodes", "arcs", "route")


class DatasetTripRequest(BaseModel):
    dataset_id: str


def _dataset_trip_files(dataset_id: str) -> dict[str, tuple[str, bytes]]:
    if not re.fullmatch(r"\d{8}", dataset_id):
        raise HTTPException(status_code=422, detail="dataset_id must be an eight-digit test dataset ID")
    folder = TEST_DATASET_DIR / dataset_id
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown test dataset_id {dataset_id}")
    files = {}
    missing = []
    for extension in TRIP_FILE_EXTENSIONS:
        path = folder / f"{dataset_id}.{extension}"
        if not path.is_file():
            missing.append(path.name)
        else:
            files[extension] = (path.name, path.read_bytes())
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"test dataset {dataset_id} is incomplete; missing: {', '.join(missing)}",
        )
    return files


def _available_test_dataset_ids() -> list[str]:
    if not TEST_DATASET_DIR.is_dir():
        return []
    return sorted(
        folder.name
        for folder in TEST_DATASET_DIR.iterdir()
        if folder.is_dir() and all((folder / f"{folder.name}.{ext}").is_file() for ext in TRIP_FILE_EXTENSIONS)
    )


def _analyze_files(files: dict[str, tuple[str, bytes]], trip_id: str) -> dict:
    try:
        trip = parse_trip(files)
    except TripValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        prediction = predict_route(trip)
    except (DependencyError, ModelLoadError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    predicted_route_segments = route_segments_for_ids(
        trip["nodes"], trip["arcs"], prediction["route_arc_ids"]
    )
    return {
        "dataset_id": trip_id,
        "gps_points": trip["gps_points"],
        "timestamps": trip["timestamps"],
        "road_segments": trip["road_segments"],
        "route_segments": trip["route_segments"],
        "route_arc_ids": trip["route_arc_ids"],
        "predicted_route_arc_ids": prediction["route_arc_ids"],
        "predicted_route_segments": predicted_route_segments,
        "model": prediction["model"],
        "is_demo": prediction["is_demo"],
        "counts": {
            "gps_observations": len(trip["gps_points"]),
            "nodes": trip["node_count"],
            "arcs": trip["arc_count"],
            "reference_route_arcs": len(trip["route_arc_ids"]),
        },
    }


@app.get("/health")
def health():
    return {"status": "ok", "mode": "xgboost_viterbi_v1"}


@app.get("/", include_in_schema=False)
def frontend():
    index = ROOT / "ui" / "index.html"
    return FileResponse(index)


@app.get("/test-datasets")
def test_datasets():
    return {"dataset_ids": _available_test_dataset_ids()}


@app.post("/trip/analyze-by-id")
def analyze_dataset_trip(request: DatasetTripRequest):
    return _analyze_files(_dataset_trip_files(request.dataset_id), request.dataset_id)


@app.post("/trip/analyze")
async def analyze_trip(
    track: UploadFile = File(...),
    nodes: UploadFile = File(...),
    arcs: UploadFile = File(...),
    route: UploadFile = File(...),
):
    uploads = {"track": track, "nodes": nodes, "arcs": arcs, "route": route}
    for kind, upload in uploads.items():
        if Path(upload.filename or "").suffix.lower() != f".{kind}":
            raise HTTPException(status_code=422, detail=f"{kind} upload must have a .{kind} filename")

    files = {}
    for kind, upload in uploads.items():
        content = await upload.read()
        files[kind] = (upload.filename or f"trip.{kind}", content)
    return _analyze_files(files, "uploaded")
