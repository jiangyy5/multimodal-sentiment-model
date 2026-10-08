# Multimodal sentiment model

## 1. Install

Follow [environment setup](docs/ENVIRONMENT.md) to create the training environment.

## 2. Prepare inputs

Follow [data and pretrained models](docs/DATA.md) for downloads, folders and preprocessing.
The dataset folder must contain `label.csv`, `Raw/`, `wav/` and `video_cache/face32/`.

## 3. Train

Run from this repository's root:

```bash
python train.py --config configs/mosi.json --seed 1
python train.py --config configs/mosei.json --seed 1
python train.py --config configs/sims.json --seed 1
```

Datasets are read from `datasets/` and pretrained assets from `pretrained_models/` by default.
To use external folders:

```bash
python train.py --config configs/mosi.json --seed 1 --data-root /path/to/datasets --models-dir /path/to/pretrained_models
```

Use `--dry-run` to inspect parameters without training, and `--num-workers 0` if multiprocessing
is unavailable (the default on Windows). Each run saves checkpoints and its configuration under `runs/`.
Training performs validation and final testing within the same run. For existing frame caches,
use `--video-cache-dir /path/to/cache`.

`configs/defaults.json` holds shared parameters; each dataset file overrides its differences.
See [training options](docs/TRAINING.md) for context, auxiliary losses and random-missing training.

## Code

- `models/`: trimodal models, fusion, CLIP and missing-input handling.
- `trainers/`: English and Chinese training routines.
- `data/`: dataset loading, preprocessing and pretrained-model download.
- `utils/`: evaluation metrics.


