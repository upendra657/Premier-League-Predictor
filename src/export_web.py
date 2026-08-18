"""Export the trained model for in-browser inference.

The served FastAPI app is the reference implementation, but a recruiter
clicking a link should not wait for a free-tier container to wake up. The
booster is small enough (171 KB gzipped) to ship to the browser and evaluate
in JavaScript, so the public demo needs no server at all.

Two files are written to ``reports/web/``:

``model.json``
    Booster topology in the minimum form an evaluator needs -- per tree, the
    left/right child arrays, split feature indices, split thresholds (leaf
    values share the threshold array, as XGBoost stores them), and the
    default-left flags for missing values.

``teams.json``
    Each team's current ratings, the same snapshot the API serves.

The thresholds are cast to float32 on the way out. XGBoost compares in
float32 internally; comparing the same values in float64 sends a feature
that sits within rounding distance of a threshold down the wrong branch.
Across 1,200 trees that happened on 17 of them and moved the output
probabilities by up to 2.6 percentage points. The browser side must call
``Math.fround`` on feature values for the same reason.
"""

from __future__ import annotations

import gzip
import json
import logging
import tempfile
from pathlib import Path

import numpy as np

from src import config
from src.features import MODEL_FEATURES

logger = logging.getLogger(__name__)

WEB_DIR = config.REPORT_DIR / "web"


def extract_booster(model) -> dict:
    """Reduce a fitted XGBoost model to the arrays an evaluator needs."""
    booster = model.get_booster()

    # save_model writes full-precision thresholds; get_dump rounds them for
    # human readability, which is not good enough to reproduce predictions.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        booster.save_model(str(path))
        raw = json.loads(path.read_text())

    learner = raw["learner"]
    params = learner["learner_model_param"]
    base_raw = params["base_score"].strip()
    n_class = int(params["num_class"])

    if base_raw.startswith("["):
        base = [float(v) for v in base_raw.strip("[]").split(",")]
    else:
        base = [float(base_raw)] * n_class

    trees = []
    for tree in learner["gradient_booster"]["model"]["trees"]:
        trees.append(
            {
                "l": [int(v) for v in tree["left_children"]],
                "r": [int(v) for v in tree["right_children"]],
                "f": [int(v) for v in tree["split_indices"]],
                # Shortest decimal that round-trips to the same float32.
                # Writing the widened float64 instead costs ~40% more bytes
                # for no accuracy: the browser calls Math.fround on both the
                # threshold and the feature before comparing, so only the
                # float32 value matters.
                "t": [float(str(np.float32(v))) for v in tree["split_conditions"]],
                "d": [int(v) for v in tree["default_left"]],
            }
        )

    return {
        "base": base,
        "n_class": n_class,
        "features": list(MODEL_FEATURES),
        "trees": trees,
    }


def main() -> dict[str, int]:
    """Write model.json and teams.json, returning their byte sizes."""
    import joblib

    WEB_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(config.MODEL_FILE)
    payload = extract_booster(bundle["model"])

    model_path = WEB_DIR / "model.json"
    blob = json.dumps(payload, separators=(",", ":"))
    model_path.write_text(blob)

    teams_path = WEB_DIR / "teams.json"
    teams_path.write_text(config.TEAM_SNAPSHOT_FILE.read_text())

    sizes = {
        "model.json": len(blob),
        "model.json.gz": len(gzip.compress(blob.encode())),
        "teams.json": teams_path.stat().st_size,
    }
    logger.info(
        "Exported %d trees (%d classes) -- %.0f KB raw, %.0f KB gzipped.",
        len(payload["trees"]),
        payload["n_class"],
        sizes["model.json"] / 1024,
        sizes["model.json.gz"] / 1024,
    )
    return sizes


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name, size in main().items():
        print(f"{name}: {size / 1024:.1f} KB")
