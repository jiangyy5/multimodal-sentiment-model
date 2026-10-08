# Environment setup

Run all commands from the repository root. Training and media preprocessing use separate environments.

## Training

The following package versions match the locally inspected Windows training environment. They are a release target, not a recovered lockfile of the original Linux/Python 3.10 experiments. A fresh installation and full GPU training have not yet been validated.

```bash
conda create -n msa-train python=3.11.14 -y
conda activate msa-train
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.is_available())"
python train.py --config configs/mosi.json --dry-run
```

The CUDA wheel requires a compatible NVIDIA driver. See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/). CPU is sufficient for configuration and synthetic checks; full training is intended for an NVIDIA GPU. GPU memory requirements have not been measured for this release layout.

On Windows use `--num-workers 0` (the launcher default). On Linux the worker count can be set explicitly with the same argument.

## Audio and video preprocessing

The source uses MoviePy 1.x and the legacy MediaPipe `solutions.face_detection` API. Use this separate Python 3.10 environment; upgrading MediaPipe without adapting the extractor is not supported.

```bash
conda create -n msa-preprocess python=3.10 -y
conda activate msa-preprocess
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-preprocess.txt
python -m pip check
python -c "import mediapipe as mp; from moviepy.editor import VideoFileClip; print(mp.solutions.face_detection)"
```

These preprocessing pins are a compatibility target and have not been installation-tested here. MoviePy uses FFmpeg via imageio-ffmpeg; if automatic binary discovery fails, set `IMAGEIO_FFMPEG_EXE` to your installed FFmpeg executable. See [data preparation](DATA.md) for commands. Return to `msa-train` before model downloads and training.

## Offline source checks

In `msa-train`, run `python -B -m unittest discover -s tests -v`. These checks cover configuration dispatch and fusion forward/backward behavior, including empty masks. They do not validate pretrained weights, real media preprocessing or full training.
