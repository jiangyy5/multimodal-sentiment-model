import os
from pathlib import Path

_MODEL_ALIASES = {
    "facebook/data2vec-audio-base": "data2vec-audio-base",
    "hfl/chinese-roberta-wwm-ext": "chinese-roberta-wwm-ext",
    "TencentGameMate/chinese-hubert-base": "chinese-hubert-base",
    "openai/clip-vit-base-patch32": "clip-vit-base-patch32-local",
}


def _model_roots():
    """Return model roots in priority order, without requiring a fixed OS path."""
    roots = []
    configured = os.environ.get("MSA_MODELS_DIR")
    if configured:
        roots.append(Path(configured).expanduser())
    source_path = Path(__file__).resolve()
    roots.append(source_path.parents[1] / "pretrained_models")
    unique = []
    seen = set()
    for root in roots:
        key = os.path.normcase(os.path.abspath(str(root)))
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_model(local_name: str, fallback: str = None) -> str:
    """Resolve a local model path if available, otherwise fall back to a HF ID."""
    if local_name is None:
        return fallback if fallback is not None else local_name
    if local_name.startswith("http"):
        return local_name
    explicit_path = Path(local_name).expanduser()
    if explicit_path.exists():
        return str(explicit_path.resolve())
    candidate_names = []
    alias = _MODEL_ALIASES.get(local_name)
    if alias:
        candidate_names.append(alias)
    candidate_names.append(local_name)
    if "/" in local_name:
        candidate_names.append(local_name.rsplit("/", 1)[-1])
    for root in _model_roots():
        for candidate_name in candidate_names:
            local_path = root / candidate_name
            if local_path.exists():
                return str(local_path.resolve())
    return fallback if fallback is not None else local_name
