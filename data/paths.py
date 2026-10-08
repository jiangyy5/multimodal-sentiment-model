import os
from pathlib import Path

_DATASET_NAMES = {
    "mosi": ("MOSI", "CMU-MOSI"),
    "mosei": ("MOSEI", "CMU-MOSEI"),
    "sims": ("SIMS", "CH-SIMS"),
}


def _unique_paths(paths):
    unique = []
    seen = set()
    for path in paths:
        path = Path(path).expanduser()
        key = os.path.normcase(os.path.abspath(str(path)))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _dataset_candidates(dataset):
    canonical_name, external_name = _DATASET_NAMES[dataset]
    source_path = Path(__file__).resolve()
    repository_root = source_path.parents[1]
    candidates = []
    dataset_env = os.environ.get(f"MSA_{dataset.upper()}_DIR")
    if dataset_env:
        candidates.append(dataset_env)
    data_root = os.environ.get("MSA_DATA_ROOT")
    if data_root:
        candidates.extend(
            (Path(data_root) / external_name, Path(data_root) / canonical_name)
        )
    candidates.extend(
        (
            repository_root / "datasets" / external_name,
            repository_root / "datasets" / canonical_name,
        )
    )
    return _unique_paths(candidates)


def resolve_dataset_dir(dataset):
    """Resolve a dataset root containing label.csv, Raw/, wav/, and video_cache/."""
    dataset = str(dataset).lower()
    if dataset not in _DATASET_NAMES:
        raise ValueError(
            f"Unsupported dataset: {dataset!r}; expected mosi, mosei, or sims"
        )
    candidates = _dataset_candidates(dataset)
    for candidate in candidates:
        if (candidate / "label.csv").is_file():
            return candidate.resolve()
    searched = "\n  - ".join((str(path) for path in candidates))
    raise FileNotFoundError(
        f"Could not locate {dataset.upper()} label.csv. Searched:\n  - {searched}\nSet MSA_{dataset.upper()}_DIR or MSA_DATA_ROOT to override."
    )


def resolve_video_cache_dir(dataset, configured=None):
    """Resolve configured legacy cache paths against the selected dataset root."""
    dataset_root = resolve_dataset_dir(dataset)
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_dir():
            return configured_path.resolve()
        parts = configured_path.parts
        lowered = [part.lower() for part in parts]
        if "video_cache" in lowered:
            suffix_start = lowered.index("video_cache")
            remapped = dataset_root.joinpath(*parts[suffix_start:])
            if remapped.is_dir():
                return remapped.resolve()
        raise FileNotFoundError(
            f"Video cache does not exist: {configured}. Resolved dataset root: {dataset_root}"
        )
    default_cache = dataset_root / "video_cache" / "face32"
    if not default_cache.is_dir():
        raise FileNotFoundError(f"Video cache does not exist: {default_cache}")
    return default_cache.resolve()
