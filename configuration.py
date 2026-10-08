"""Parameter loading shared by training and lightweight configuration checks."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_configuration(path):
    defaults = json.loads((ROOT / "configs/defaults.json").read_text(encoding="utf-8"))
    overrides = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(overrides, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    parameters = dict(defaults, **overrides)
    for key, value in parameters.items():
        expected = type(defaults[key])
        valid = type(value) is expected
        if expected is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        if not valid:
            raise ValueError(f"Invalid value type for {key}")
    if parameters["dataset"] not in ("mosi", "mosei", "sims"):
        raise ValueError("dataset must be mosi, mosei or sims")
    if parameters["video_model"] not in ("clip_vitb32", "openai/clip-vit-base-patch32"):
        raise ValueError("This implementation uses OpenAI CLIP ViT-B/32")
    for key in (
        "batch_size",
        "grad_accum_steps",
        "epochs",
        "early_stop",
        "num_hidden_layers",
        "video_frames",
    ):
        if parameters[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if parameters["lr"] <= 0:
        raise ValueError("lr must be positive")
    for key in ("tasks", "denoise_tasks"):
        tasks = parameters[key]
        if (
            not tasks
            or any(t not in "MTAV" for t in tasks)
            or len(set(tasks)) != len(tasks)
        ):
            raise ValueError(f"{key} must contain distinct M/T/A/V task letters")
    if not parameters["tasks"].startswith("M"):
        raise ValueError(
            "tasks must start with M for fused-branch checkpoint selection"
        )
    if parameters["dataset"] == "sims" and parameters["context"]:
        raise ValueError("SIMS uses no context; set context to false")
    return parameters
