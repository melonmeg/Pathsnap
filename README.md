# PathSnap - distributed map-matching & route reconstruction

Noisy 1 Hz GPS fixes -> road-network arcs. Pipeline:

1. **Candidate generation** - per GPS fix, all arcs within 50 m (max 12 nearest; fallback to nearest arcs if fewer than 3).
2. **Candidate classifier** - every (fix, candidate arc) row is scored with geometric/contextual features
   (distance, heading agreement, along-arc position, distance of neighbouring fixes to the same arc, progress direction, ...).
   Arc ids are *not* features (each trip has its own network).
3. **HMM / Viterbi** - hidden state = candidate arc of each fix; emission = classifier probability; transition = |GPS distance - network route distance|.
4. **FastAPI** server (Laptop 1) + **static HTML/Leaflet UI** (Laptop 2).

```
src/pathsnap/   shared package (io, network, features, labels, hmm, metrics, models, pipeline, evaluate)
scripts/        01_eda, 02_build_features, 03_train_compare, 04_bayes_opt, 05_hmm_tune_eval,
                06_folium_maps, 07_plots, 08_export_xgboost
api/main.py     FastAPI backend          ui/index.html   evaluator UI
artifacts/      model.joblib (XGBoost, served model), model_meta.json, hmm_config.json
results/        CSV results, experiment logs     figures/   plots + Folium HTML maps
notebooks/colab_run_all.ipynb   runs the whole experiment on Colab
```

## Served model
**XGBoost** (`artifacts/model.joblib`), trained on the 73 usable training trips with the configuration registered as `XGBoost`
in `src/pathsnap/models.py`, decoded by the HMM with the settings in `artifacts/hmm_config.json`.

## Model selection
Candidates were compared with 5-fold `GroupKFold` CV on the training trips, then reported on the student-test trips.
Test-set numbers were used to make the final choice among the three boosted models, which are close on CV, so the choice is a judgement call.

| Model | CV macro F1 | CV log loss | Test macro F1 | Test top-1 | Test log loss | Test route continuity (strict) |
|---|---|---|---|---|---|---|
| LightGBM (untuned) | 0.9382 | 0.0771 | 0.9353 | 0.8917 | 0.0766 | 0.787 |
| Random Forest | 0.9389 | 0.0716 | 0.9351 | 0.8873 | 0.0695 | 0.831 |
| **XGBoost (chosen)** | **0.9392** | **0.0717** | **0.9385** | **0.8974** | **0.0664** | 0.825 |

Reasons for XGBoost:
* CV macro F1 is level with the other boosted models (within noise).
* Log loss is as good as Random Forest on CV and best on test. The HMM uses classifier probabilities as emission scores, so calibration matters.
* Best test macro F1 and top-1 of the three.

After HMM decoding the difference disappears. On the student-test trips, LightGBM + HMM gives top-1 0.9394 and route F1 0.9696, and XGBoost + HMM gives 0.9391 and 0.9690.
The choice was made on classifier-level evidence, and the final system is equivalent under either model.

Other models (MLP, Naive Bayes, Logistic Regression, the nearest-arc baseline) were run earlier and are weaker or no better than the boosted trees.
Naive Bayes is weakest: its features are correlated, so its probabilities are overconfident (test log loss 0.91). It's kept as a baseline only.

**Not yet run:** ExtraTrees and HistGradientBoosting are defined in `models.py` but have no results.

**Tuning:** LightGBM was tuned with Optuna TPE for **6 trials** (`results/bayes_best_params.json`), not the 30 used in an earlier version of this README.
The gain was negligible (CV macro F1 0.9382 -> 0.9390, within the fold standard deviation of about 0.018), so the tuned settings were not kept as the served model.

## Data-quality decisions (see `results/trip_stats_train.csv`)
Seven training trips have a reference route that does not describe the recorded drive, and they are excluded from fitting and CV:
trips 00000002, 00000033, 00000043, 00000054, 00000061, 00000066, 00000072.
Trips are excluded when mean label distance > 15 m or candidate-miss rate > 5 % (`config.QC_*`). Trip 00000061 is excluded for its miss rate (6.6 %), not its label distance.
Individual fixes whose reference arc is outside the candidate set are not used as supervision.
Student-test trips are never filtered, so CV scores (filtered trips) and test scores (unfiltered) are not directly comparable.
Evaluator-test data is never used for any choice.

## Definitions (used consistently everywhere)
* **Reference label per fix**: `.route` is only an ordered arc list, so each fix is aligned to the route with a monotone
  dynamic-programming alignment (`labels.py`) minimising capped fix-to-arc distance. The same function produces training labels,
  student-test scores and the API's server-side score. Reference routes are treated as *reference*, not verified truth.
* **Macro F1 / Precision / Recall / Log loss**: candidate-row binary classification (y=1 iff row's arc is the reference arc); macro = mean over both classes, threshold 0.5.
* **Top-1 accuracy**: per fix, highest-scoring candidate == reference arc; fixes whose reference arc is not a candidate count as wrong.
* **Candidate miss rate**: fixes whose reference arc is absent from the candidate set.
* **Route continuity**: of consecutive fixes with different predicted arcs, share that are network-connected
  (*strict*: next arc starts where previous ends; *relaxed*: reachable within 2*g+150 m).
* **Route F1**: precision/recall/F1 of the arc set of the reconstructed (gap-filled) route vs the reference route.
* **CV**: 5-fold `GroupKFold` grouped by trip on the 73 usable training trips (no trip split across folds).

## Run the experiments (Colab recommended)
Open `notebooks/colab_run_all.ipynb`, or locally:
```
pip install -r requirements.txt
python scripts/01_eda.py              --data-root <dir with train/ and student_test/>
python scripts/02_build_features.py   --data-root <...>
python scripts/03_train_compare.py    --data-root <...>                       # all models + nearest-arc baseline
python scripts/03_train_compare.py    --data-root <...> --models XGBoost      # or a single model; keeps other rows
python scripts/04_bayes_opt.py        --data-root <...> --trials 30           # optional LightGBM tuning (Optuna TPE)
python scripts/05_hmm_tune_eval.py    --data-root <...> --model XGBoost --oof oof_probs.pkl --student student_probs.pkl
python scripts/08_export_xgboost.py                                           # trains the served XGBoost, writes artifacts/model.joblib
python scripts/07_plots.py ; python scripts/06_folium_maps.py --data-root <...> --trip <id>
```
Script 05 writes `artifacts/hmm_config.json` and results tagged by `--tag`. Decoder parameters are tuned on out-of-fold *training* predictions only.
If the served model changes, re-run 05 for that model and then 08 for the export, and update this README's model table.

Note: scikit-learn tree models need a working `sklearn.tree` DLL (some locked-down Windows machines block it). The served XGBoost model does not need sklearn.

## Deployment (two laptops)
**Laptop 1 (API)** - copy repo + the folder of the 10 reserved trips (each trip folder with `.track .nodes .arcs .route`):
```
pip install -r requirements.txt
set PATHSNAP_TRIPS_DIR=D:\evaluator_trips        (PowerShell: $env:PATHSNAP_TRIPS_DIR="D:\evaluator_trips")
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Allow TCP 8000 in Windows Firewall; find the IP with `ipconfig`. Every request prints trip id, metrics and timings to this console (show it to evaluators).
Endpoints: `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`. Reference routes stay server-side; responses contain only predictions and scores.

**Laptop 2 (UI)**:
```
python -m http.server 8080 --directory ui
```
open `http://localhost:8080`, enter `http://<laptop1-ip>:8000`, click *Load trips*, then a trip. (Leaflet/OSM tiles load from the internet; for offline
use copy Leaflet 1.9.4 into `ui/vendor/leaflet/` - route/GPS layers draw without tiles.)

## AI-assistance disclosure
This code base was drafted with an AI coding assistant (Claude Code). All numbers in reports must come from running the scripts; the
team is responsible for understanding and being able to explain every component.
