"""Download pretrained assets with an explicitly recorded Hugging Face revision."""

import argparse
import json
from pathlib import Path

ASSETS = {
    "roberta-base": ("FacebookAI/roberta-base", False),
    "roberta-large": ("FacebookAI/roberta-large", True),
    "data2vec-audio-base": ("facebook/data2vec-audio-base", False),
    "chinese-roberta-wwm-ext": ("hfl/chinese-roberta-wwm-ext", False),
    "chinese-hubert-base": ("TencentGameMate/chinese-hubert-base", False),
    "clip-vit-base-patch32-local": ("openai/clip-vit-base-patch32", False),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("mosi", "mosei", "sims", "all"), default="all"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pretrained_models",
    )
    parser.add_argument(
        "--revisions",
        type=Path,
        help="JSON object mapping model IDs to desired commit hashes",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    english = {"roberta-base", "roberta-large", "data2vec-audio-base"}
    chinese = {"chinese-roberta-wwm-ext", "chinese-hubert-base"}
    names = (
        english | chinese
        if args.dataset == "all"
        else chinese if args.dataset == "sims" else english
    )
    names = names | {"clip-vit-base-patch32-local"}
    requested = (
        json.loads(args.revisions.read_text(encoding="utf-8")) if args.revisions else {}
    )
    if args.dry_run:
        print(json.dumps({name: ASSETS[name][0] for name in sorted(names)}, indent=2))
        return
    from huggingface_hub import HfApi, snapshot_download

    args.output_dir.mkdir(parents=True, exist_ok=True)
    record_path = args.output_dir / "revisions.json"
    recorded = (
        json.loads(record_path.read_text(encoding="utf-8"))
        if record_path.exists()
        else {}
    )
    api = HfApi()
    for name in sorted(names):
        repo_id, tokenizer_only = ASSETS[name]
        # Reuse a previously resolved commit instead of updating existing assets silently.
        revision = requested.get(repo_id, recorded.get(repo_id, "main"))
        commit = api.model_info(repo_id, revision=revision).sha
        patterns = ["*.json", "*.txt", "*.model", "merges.txt", "vocab.*"]
        if not tokenizer_only:
            patterns += ["*.bin", "*.safetensors"]
        snapshot_download(
            repo_id,
            revision=commit,
            local_dir=str(args.output_dir / name),
            allow_patterns=patterns,
        )
        recorded[repo_id] = commit
        record_path.write_text(json.dumps(recorded, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {commit}")


if __name__ == "__main__":
    main()
