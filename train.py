"""Train one dataset/seed using a JSON configuration."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from configuration import load_configuration

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--video-cache-dir", type=Path)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parameters without loading models or training",
    )
    cli = parser.parse_args()
    try:
        parameters = load_configuration(cli.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if cli.seed < 0:
        parser.error("--seed must be nonnegative")
    parameters["seed"] = cli.seed
    if cli.video_cache_dir:
        parameters["video_cache_dir"] = str(cli.video_cache_dir.resolve())
    for key, value in (
        ("MSA_DATA_ROOT", cli.data_root),
        ("MSA_MODELS_DIR", cli.models_dir),
    ):
        if value is not None:
            os.environ[key] = str(value.resolve())
    if cli.num_workers is not None:
        if cli.num_workers < 0:
            parser.error("--num-workers must be nonnegative")
        os.environ["MSA_NUM_WORKERS"] = str(cli.num_workers)
    elif os.name == "nt":
        os.environ.setdefault("MSA_NUM_WORKERS", "0")
    if cli.dry_run:
        print(json.dumps(parameters, indent=2))
        return
    # Resolve data before allocating pretrained models or creating run files.
    from data.paths import resolve_dataset_dir, resolve_video_cache_dir

    resolve_dataset_dir(parameters["dataset"])
    parameters["video_cache_dir"] = str(
        resolve_video_cache_dir(parameters["dataset"], parameters["video_cache_dir"])
    )
    from trainers.run import main as run_training

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    output = ROOT / "runs" / parameters["dataset"] / f"seed{cli.seed}-{run_id}"
    output.mkdir(parents=True, exist_ok=False)
    parameters["model_save_path"] = str(output / "checkpoints")
    (output / "config.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Run directory: {output}", flush=True)
    run_training(argparse.Namespace(**parameters))


if __name__ == "__main__":
    main()
