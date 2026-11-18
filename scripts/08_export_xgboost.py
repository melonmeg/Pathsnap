"""Train the final XGBoost candidate scorer on the full filtered training partition and export it for the API.

Uses exactly the configuration registered as "XGBoost" in src/pathsnap/models.py, so the exported model is the
one that was cross-validated in 03_train_compare.py. Student-test data is not used here.

Usage: python scripts/08_export_xgboost.py [--results results_rf]
Outputs: artifacts/model.joblib, artifacts/model_meta.json (kind = joblib)
"""
import argparse
import json
import os
import sys

import joblib
import xgboost

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.dataset import load_partition, xy_of  # noqa: E402
from pathsnap.models import make_models  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=C.RESULT_DIR)
    ap.add_argument("--artifact-dir", default=C.ARTIFACT_DIR)
    args = ap.parse_args()

    tr, _ = load_partition("train", args.results, training=True)
    X, y = xy_of(tr)
    print(f"training rows={len(tr)} trips={tr['trip'].nunique()}")

    factory = make_models()["XGBoost"][2]
    model = factory()
    model.fit(X, y)

    os.makedirs(args.artifact_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.artifact_dir, "model.joblib"))
    json.dump(dict(kind="joblib", file="model.joblib", features=C.FEATURES, algorithm="XGBoost",
                   config=make_models()["XGBoost"][1], xgboost_version=xgboost.__version__,
                   radius=C.CAND_RADIUS_M, max_k=C.MAX_CANDIDATES),
              open(os.path.join(args.artifact_dir, "model_meta.json"), "w"), indent=2)
    print("exported XGBoost model to", args.artifact_dir)


if __name__ == "__main__":
    main()
