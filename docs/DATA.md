# Data and pretrained models

## Download datasets

Dataset project references: [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) and [MMSA / CH-SIMS](https://github.com/thuiar/MMSA). Follow the data providers' access and usage terms. This repository does not redistribute datasets.

Use raw utterance videos and CSV labels in the format below. Precomputed feature pickles are not accepted by this loader. Keep the provider's train/valid/test assignments.

```text
datasets/
  CMU-MOSI/
    label.csv
    Raw/<video_id>/<clip_id>.mp4
    wav/<video_id>/<clip_id>.wav
    video_cache/face32/<video_id>/<clip_id>.pt
  CMU-MOSEI/             # same structure
  CH-SIMS/              # same structure
```

`MOSI`, `MOSEI`, `SIMS` are also accepted folder names. Use the exact `Raw` capitalization on Linux. SIMS clip filenames normally use four-digit zero padding. CSV fields are `video_id`, `clip_id`, `text`, `label`, `mode`; SIMS additionally uses `label_T`, `label_A`, `label_V`, `polarity`. Split values are `train`, `valid`, `test`.

For external data set `MSA_DATA_ROOT` to the parent of the dataset folders. In PowerShell: `$env:MSA_DATA_ROOT = "E:/datasets"`; in Bash: `export MSA_DATA_ROOT=/path/to/datasets`. Per-dataset variables `MSA_MOSI_DIR`, `MSA_MOSEI_DIR`, `MSA_SIMS_DIR` take priority and point directly to the folder containing `label.csv`.

## Preprocess raw videos

Activate the preprocessing environment from [environment setup](ENVIRONMENT.md). Run from the repository root:

```bash
python -m data.extract_audio --dataset mosi
python -m data.extract_video_cache --dataset mosi --frames 32 --size 224 --face_margin 0.2 --min_conf 0.5 --skip_existing
```

Repeat with `mosei` and `sims` as needed. Video extraction also accepts `--csv_path`, `--video_root`, `--output_root`. Extraction writes the `wav/` and `video_cache/face32/` folders under the selected dataset.

Audio is 16 kHz mono PCM; the loader truncates each clip to 96,000 samples. Video extraction uniformly samples 32 frames, selects the largest face with a margin and resizes to 224 × 224. A missing face uses the previous box or full image. Failed frame reads receive a false mask. Caches contain RGB pixels and masks, not precomputed backbone embeddings.

English loading sorts clips, filters the invalid-audio IDs listed in `data/dataset.py`, and forms context within the same split/video. SIMS duplicate handling differs between the complete and missing-input trainers; see [training options](TRAINING.md).

## Download pretrained assets

Activate `msa-train`. These are the exact upstream model variants used by the loaders:

| Local folder under `pretrained_models/` | Upstream source | Usage |
|---|---|---|
| `roberta-base` | [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) | English text backbone |
| `roberta-large` | [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) | English tokenizer only |
| `data2vec-audio-base` | [facebook/data2vec-audio-base](https://huggingface.co/facebook/data2vec-audio-base) | English audio |
| `chinese-roberta-wwm-ext` | [hfl/chinese-roberta-wwm-ext](https://huggingface.co/hfl/chinese-roberta-wwm-ext) | Chinese text |
| `chinese-hubert-base` | [TencentGameMate/chinese-hubert-base](https://huggingface.co/TencentGameMate/chinese-hubert-base) | Chinese audio |
| `clip-vit-base-patch32-local` | [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) | Visual backbone and processor |

```bash
python -m data.download_models --dataset all --dry-run
python -m data.download_models --dataset all
```

Use `--dataset mosi`, `mosei` or `sims` for a subset, and `--output-dir /path/to/pretrained_models` for external storage. Point training to that directory using `--models-dir` or `MSA_MODELS_DIR`.

The downloader resolves Hugging Face revisions to commit hashes and records them in `pretrained_models/revisions.json`; subsequent downloads reuse those hashes. To select explicit versions, pass `--revisions revisions.json`, a JSON object mapping upstream repository IDs to commit hashes. Historical experiment commit hashes were not available in this snapshot; a newly recorded revision must not be presented as their recovered version.

The visual encoder uses the original OpenAI CLIP ViT-B/32 pretrained weights. Include its processor configuration as well as weights. The download command obtains model configurations, tokenizer/processor files and weights; `roberta-large` only needs tokenizer files. Datasets, downloaded weights, generated caches and local training outputs are excluded by `.gitignore`.
