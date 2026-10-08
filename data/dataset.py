import os
import torch
import soundfile as sf
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models.paths import resolve_model
from data.paths import resolve_dataset_dir, resolve_video_cache_dir

SIMS_DUPLICATE_TRAIN_KEYS = {("video_0049", 8)}


def _load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 1:
        audio = audio[None, :]
    else:
        audio = audio.T
    return (torch.from_numpy(audio), sr)


class Dataset_sims(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path,
        audio_directory,
        mode,
        video_cache_dir=None,
        video_frames=32,
        drop_known_cross_split_duplicates=False,
    ):
        df = pd.read_csv(csv_path)
        if "mode" not in df.columns:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=[
                    "video_id",
                    "clip_id",
                    "text",
                    "label",
                    "label_T",
                    "label_A",
                    "label_V",
                    "polarity",
                    "mode",
                ],
            )
        df = df[df["mode"] == mode].copy()
        if mode == "train" and drop_known_cross_split_duplicates:
            clip_numbers = pd.to_numeric(df["clip_id"], errors="coerce")
            duplicate_mask = pd.Series(False, index=df.index)
            for video_id, clip_id in SIMS_DUPLICATE_TRAIN_KEYS:
                duplicate_mask |= df["video_id"].astype(str).eq(
                    video_id
                ) & clip_numbers.eq(clip_id)
            removed = int(duplicate_mask.sum())
            df = df.loc[~duplicate_mask]
            if removed:
                print(
                    f"CH-SIMS leakage audit: removed {removed} known train/test duplicate from the training split."
                )
        df = df.reset_index(drop=True)
        self.targets_M = df["label"]
        self.targets_T = df["label_T"]
        self.targets_A = df["label_A"]
        self.targets_V = df["label_V"]
        self.texts = df["text"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolve_model("chinese-roberta-wwm-ext", "hfl/chinese-roberta-wwm-ext")
        )
        self.audio_file_paths = []
        for i in range(0, len(df)):
            clip_id = str(df["clip_id"][i])
            for j in range(4 - len(clip_id)):
                clip_id = "0" + clip_id
            file_name = str(df["video_id"][i]) + "/" + clip_id + ".wav"
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.video_cache_dir = video_cache_dir
        self.video_frames = video_frames
        if not self.video_cache_dir:
            raise ValueError("video_cache_dir must be set when use_video=True")
        self.video_cache_paths = []
        for i in range(len(df)):
            clip_id = str(df["clip_id"][i])
            if clip_id.isdigit():
                clip_id = f"{int(clip_id):04d}"
            file_path = os.path.join(
                self.video_cache_dir, str(df["video_id"][i]), f"{clip_id}.pt"
            )
            self.video_cache_paths.append(file_path)

    def _fit_video_seq_len(self, frames, mask):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        frames = frames.to(dtype=torch.float32)
        if mask is None:
            mask = torch.ones(frames.shape[0], dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool).flatten()
            if mask.shape[0] < frames.shape[0]:
                pad = frames.shape[0] - mask.shape[0]
                mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool)], dim=0)
            elif mask.shape[0] > frames.shape[0]:
                mask = mask[: frames.shape[0]]
        target_len = int(self.video_frames)
        if frames.shape[0] == 0:
            frames = torch.zeros((target_len, 3, 224, 224), dtype=torch.float32)
            mask = torch.zeros(target_len, dtype=torch.bool)
            return (frames, mask)
        if frames.shape[0] < target_len:
            pad = target_len - frames.shape[0]
            pad_frames = frames[-1:].repeat(pad, 1, 1, 1)
            frames = torch.cat([frames, pad_frames], dim=0)
            mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool)], dim=0)
        elif frames.shape[0] > target_len:
            frames = frames[:target_len]
            mask = mask[:target_len]
        return (frames, mask)

    def _load_video_clip(self, index):
        cache_path = self.video_cache_paths[index]
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing video cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        if isinstance(cache, dict):
            frames = cache.get("frames", None)
            mask = cache.get("mask", None)
        else:
            frames = cache
            mask = None
        if frames is None:
            raise KeyError(f"Video cache missing 'frames': {cache_path}")
        return self._fit_video_seq_len(frames, mask)

    def __getitem__(self, index):
        text = str(self.texts[index])
        tokenized_text = self.tokenizer(
            text,
            max_length=64,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        sound, _ = _load_audio(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)
        features = self.feature_extractor(
            soundData,
            sampling_rate=16000,
            max_length=96000,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
        )
        audio_features = torch.tensor(
            np.array(features["input_values"]), dtype=torch.float32
        ).squeeze()
        audio_masks = torch.tensor(
            np.array(features["attention_mask"]), dtype=torch.long
        ).squeeze()
        sample = {
            "text_tokens": tokenized_text["input_ids"],
            "text_masks": tokenized_text["attention_mask"],
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "target": {
                "M": self.targets_M[index],
                "T": self.targets_T[index],
                "A": self.targets_A[index],
            },
        }
        frames, mask = self._load_video_clip(index)
        sample["video_inputs"] = frames
        sample["video_masks"] = mask
        sample["target"]["V"] = self.targets_V[index]
        return sample

    def __len__(self):
        return len(self.targets_M)


class Dataset_mosi(torch.utils.data.Dataset):

    def __init__(
        self,
        csv_path,
        audio_directory,
        mode,
        text_context_length=2,
        audio_context_length=1,
        video_cache_dir=None,
        video_frames=32,
    ):
        df = pd.read_csv(csv_path)
        invalid_files = [
            "3aIQUQgawaI/12.wav",
            "94ULum9MYX0/2.wav",
            "mRnEJOLkhp8/24.wav",
            "aE-X_QdDaqQ/3.wav",
            "94ULum9MYX0/11.wav",
            "mRnEJOLkhp8/26.wav",
        ]
        for f in invalid_files:
            video_id = f.split("/")[0]
            clip_id = f.split("/")[1].split(".")[0]
            df = df[~((df["video_id"] == video_id) & (df["clip_id"] == int(clip_id)))]
        df = (
            df[df["mode"] == mode].sort_values(by=["video_id", "clip_id"]).reset_index()
        )
        self.targets_M = df["label"]
        df["text"] = df["text"].str[0] + df["text"].str[1:].apply(lambda x: x.lower())
        self.texts = df["text"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolve_model("roberta-large", "roberta-large")
        )
        self.audio_file_paths = []
        for i in range(0, len(df)):
            file_name = str(df["video_id"][i]) + "/" + str(df["clip_id"][i]) + ".wav"
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        self.video_id = df["video_id"]
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length
        self.video_cache_dir = video_cache_dir
        self.video_frames = video_frames
        if not self.video_cache_dir:
            raise ValueError("video_cache_dir must be set when use_video=True")
        self.video_cache_paths = []
        for i in range(0, len(df)):
            clip_id = str(df["clip_id"][i])
            file_name = f"{clip_id}.pt"
            file_path = os.path.join(
                self.video_cache_dir, str(df["video_id"][i]), file_name
            )
            self.video_cache_paths.append(file_path)

    def _fit_video_seq_len(self, frames, mask):
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if not torch.is_tensor(frames):
            raise TypeError("video frames must be a tensor or numpy array")
        frames = frames.to(dtype=torch.float32)
        if frames.ndim != 4:
            raise ValueError(
                f"video frames must be [T,C,H,W], got shape={tuple(frames.shape)}"
            )
        if mask is None:
            mask = torch.ones(frames.shape[0], dtype=torch.bool)
        else:
            mask = mask.to(dtype=torch.bool).flatten()
            if mask.shape[0] < frames.shape[0]:
                pad = frames.shape[0] - mask.shape[0]
                mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool)], dim=0)
            elif mask.shape[0] > frames.shape[0]:
                mask = mask[: frames.shape[0]]
        target_len = int(self.video_frames)
        if frames.shape[0] == 0:
            frames = torch.zeros((target_len, 3, 224, 224), dtype=torch.float32)
            mask = torch.zeros(target_len, dtype=torch.bool)
            return (frames, mask)
        if frames.shape[0] < target_len:
            pad = target_len - frames.shape[0]
            pad_frames = frames[-1:].repeat(pad, 1, 1, 1)
            frames = torch.cat([frames, pad_frames], dim=0)
            mask = torch.cat([mask, torch.zeros(pad, dtype=torch.bool)], dim=0)
        elif frames.shape[0] > target_len:
            frames = frames[:target_len]
            mask = mask[:target_len]
        return (frames, mask)

    def _load_video_clip(self, index):
        cache_path = self.video_cache_paths[index]
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Missing video cache: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        if isinstance(cache, dict):
            frames = cache.get("frames", None)
            mask = cache.get("mask", None)
        else:
            frames = cache
            mask = None
        if frames is None:
            raise KeyError(f"Video cache missing 'frames': {cache_path}")
        return self._fit_video_seq_len(frames, mask)

    def __getitem__(self, index):
        text = str(self.texts[index])
        text_context = ""
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts[index - i])
                text_context = context + "</s>" + text_context
        tokenized_text = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        text_context = text_context[:-4]
        tokenized_context = self.tokenizer(
            text_context,
            max_length=96,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        frames, mask = self._load_video_clip(index)
        video_context_frames = []
        video_context_masks = []
        for i in range(1, self.audio_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            context_frames, context_mask = self._load_video_clip(index - i)
            video_context_frames.insert(0, context_frames)
            video_context_masks.insert(0, context_mask)
        if len(video_context_frames) == 0:
            context_frames = torch.zeros_like(frames)
            context_mask = torch.zeros_like(mask)
        else:
            context_frames = torch.cat(video_context_frames, dim=0)
            context_mask = torch.cat(video_context_masks, dim=0)
            context_frames, context_mask = self._fit_video_seq_len(
                context_frames, context_mask
            )
        sound, _ = _load_audio(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)
        audio_context = torch.tensor([])
        for i in range(1, self.audio_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context, _ = _load_audio(self.audio_file_paths[index - i])
                contextData = torch.mean(context, dim=0, keepdim=False)
                audio_context = torch.cat((contextData, audio_context), 0)
        features = self.feature_extractor(
            soundData,
            sampling_rate=16000,
            max_length=96000,
            return_attention_mask=True,
            truncation=True,
            padding="max_length",
        )
        audio_features = torch.tensor(
            np.array(features["input_values"]), dtype=torch.float32
        ).squeeze()
        audio_masks = torch.tensor(
            np.array(features["attention_mask"]), dtype=torch.long
        ).squeeze()
        if len(audio_context) == 0:
            audio_context_features = torch.zeros(96000, dtype=torch.float32)
            audio_context_masks = torch.zeros(96000, dtype=torch.long)
        else:
            features = self.feature_extractor(
                audio_context,
                sampling_rate=16000,
                max_length=96000,
                return_attention_mask=True,
                truncation=True,
                padding="max_length",
            )
            audio_context_features = torch.tensor(
                np.array(features["input_values"]), dtype=torch.float32
            ).squeeze()
            audio_context_masks = torch.tensor(
                np.array(features["attention_mask"]), dtype=torch.long
            ).squeeze()
        return {
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(
                tokenized_text["attention_mask"], dtype=torch.long
            ),
            "text_context_tokens": torch.tensor(
                tokenized_context["input_ids"], dtype=torch.long
            ),
            "text_context_masks": torch.tensor(
                tokenized_context["attention_mask"], dtype=torch.long
            ),
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "audio_context_inputs": audio_context_features,
            "audio_context_masks": audio_context_masks,
            "video_inputs": frames,
            "video_masks": mask,
            "video_context_inputs": context_frames,
            "video_context_masks": context_mask,
            "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
        }

    def __len__(self):
        return len(self.targets_M)


def collate_fn_sims(batch):
    text_tokens = []
    text_masks = []
    audio_inputs = []
    audio_masks = []
    video_inputs = []
    video_masks = []
    targets_M = []
    targets_T = []
    targets_A = []
    targets_V = []
    has_video = "video_inputs" in batch[0]
    for i in range(len(batch)):
        text_tokens.append(batch[i]["text_tokens"])
        text_masks.append(batch[i]["text_masks"])
        audio_inputs.append(batch[i]["audio_inputs"])
        audio_masks.append(batch[i]["audio_masks"])
        if has_video:
            video_inputs.append(batch[i]["video_inputs"])
            video_masks.append(batch[i]["video_masks"])
        targets_M.append(batch[i]["target"]["M"])
        targets_T.append(batch[i]["target"]["T"])
        targets_A.append(batch[i]["target"]["A"])
        if has_video:
            targets_V.append(batch[i]["target"]["V"])
    collated = {
        "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
        "text_masks": torch.tensor(text_masks, dtype=torch.long),
        "audio_inputs": torch.stack(audio_inputs),
        "audio_masks": torch.stack(audio_masks),
        "targets": {
            "M": torch.tensor(targets_M, dtype=torch.float32),
            "T": torch.tensor(targets_T, dtype=torch.float32),
            "A": torch.tensor(targets_A, dtype=torch.float32),
        },
    }
    if has_video:
        collated["video_inputs"] = torch.stack(video_inputs)
        collated["video_masks"] = torch.stack(video_masks)
        collated["targets"]["V"] = torch.tensor(targets_V, dtype=torch.float32)
    return collated


def _build_loader(dataset, batch_size, shuffle, collate_fn=None):
    env_workers = os.getenv("MSA_NUM_WORKERS")
    if env_workers is not None:
        try:
            num_workers = max(0, int(env_workers))
        except ValueError:
            num_workers = min(4, True)
    else:
        num_workers = min(4, True)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    try:
        return DataLoader(dataset, **loader_kwargs)
    except PermissionError:
        loader_kwargs["num_workers"] = 0
        loader_kwargs.pop("persistent_workers", None)
        return DataLoader(dataset, **loader_kwargs)


def data_loader(
    batch_size,
    dataset,
    text_context_length=2,
    audio_context_length=1,
    video_cache_dir=None,
    video_frames=32,
    drop_sims_train_test_duplicate=False,
):
    dataset = str(dataset).lower()
    dataset_root = resolve_dataset_dir(dataset)
    csv_path = str(dataset_root / "label.csv")
    audio_file_path = str(dataset_root / "wav")
    video_cache_dir = str(resolve_video_cache_dir(dataset, video_cache_dir))
    if dataset == "mosi":
        train_data = Dataset_mosi(
            csv_path,
            audio_file_path,
            "train",
            text_context_length=text_context_length,
            audio_context_length=audio_context_length,
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        test_data = Dataset_mosi(
            csv_path,
            audio_file_path,
            "test",
            text_context_length=text_context_length,
            audio_context_length=audio_context_length,
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        val_data = Dataset_mosi(
            csv_path,
            audio_file_path,
            "valid",
            text_context_length=text_context_length,
            audio_context_length=audio_context_length,
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        train_loader = _build_loader(train_data, batch_size, shuffle=True)
        test_loader = _build_loader(test_data, batch_size, shuffle=False)
        val_loader = _build_loader(val_data, batch_size, shuffle=False)
        return (train_loader, test_loader, val_loader)
    elif dataset == "mosei":
        train_data = Dataset_mosi(
            csv_path,
            audio_file_path,
            "train",
            text_context_length=text_context_length,
            audio_context_length=audio_context_length,
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        test_data = Dataset_mosi(
            csv_path,
            audio_file_path,
            "test",
            text_context_length=text_context_length,
            audio_context_length=audio_context_length,
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        val_data = Dataset_mosi(
            csv_path,
            audio_file_path,
            "valid",
            text_context_length=text_context_length,
            audio_context_length=audio_context_length,
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        train_loader = _build_loader(train_data, batch_size, shuffle=True)
        test_loader = _build_loader(test_data, batch_size, shuffle=False)
        val_loader = _build_loader(val_data, batch_size, shuffle=False)
        return (train_loader, test_loader, val_loader)
    elif dataset == "sims":
        train_data = Dataset_sims(
            csv_path,
            audio_file_path,
            "train",
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
            drop_known_cross_split_duplicates=drop_sims_train_test_duplicate,
        )
        test_data = Dataset_sims(
            csv_path,
            audio_file_path,
            "test",
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        val_data = Dataset_sims(
            csv_path,
            audio_file_path,
            "valid",
            video_cache_dir=video_cache_dir,
            video_frames=video_frames,
        )
        train_loader = _build_loader(
            train_data, batch_size, shuffle=True, collate_fn=collate_fn_sims
        )
        test_loader = _build_loader(
            test_data, batch_size, shuffle=False, collate_fn=collate_fn_sims
        )
        val_loader = _build_loader(
            val_data, batch_size, shuffle=False, collate_fn=collate_fn_sims
        )
        return (train_loader, test_loader, val_loader)
    raise ValueError(f"Unsupported dataset: {dataset!r}; expected mosi, mosei, or sims")
