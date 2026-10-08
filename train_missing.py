"""Evaluate trimodal checkpoints under LNLN-style random local missingness.

The corruption is applied at the model inputs without changing checkpoint
parameters:

* text: replace randomly selected non-special tokens with the tokenizer UNK id;
* audio: zero projected convolutional time steps before audio self-attention;
* video: zero CLIP frame features before video temporal self-attention.

The original validity masks remain unchanged for audio/video, so missingness is
represented by zero features as in LNLN rather than being disclosed through a
new attention mask.

Masks use LNLN's per-position Bernoulli sampling. They are deterministic,
nested across missing rates, and independent of batch size. The same
``--mask_seed`` therefore produces directly comparable test corruptions for
different checkpoints and ablations.
"""

import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from models.chinese import ChineseModel
from trainers.chinese import ChConfig, ChTrainer
from models.english import EnglishModel
from data.dataset import data_loader
from trainers.english import EnConfig, EnTrainer
from utils.metrics import MetricsTop
from models.missing import select_bernoulli_subset

DEFAULT_RATES = [i / 10 for i in range(10)]
VALID_MODES = ("all", "text", "audio", "video")
KEY_SEED_OFFSETS = {
    "text_tokens": 11,
    "text_context_tokens": 13,
    "audio_inputs": 17,
    "audio_context_inputs": 19,
    "video_inputs": 23,
    "video_context_inputs": 29,
}
ENGLISH_METRICS = [
    "Has0_acc_2",
    "Has0_F1_score",
    "Non0_acc_2",
    "Non0_F1_score",
    "Mult_acc_5",
    "Mult_acc_7",
    "MAE",
    "Corr",
    "Loss",
]
SIMS_METRICS = [
    "Mult_acc_2",
    "Mult_acc_3",
    "Mult_acc_5",
    "F1_score",
    "MAE",
    "Corr",
    "Loss",
]


def parse_float_list(value):
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "rates must be comma-separated floats"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one missing rate is required")
    if any((rate < 0.0 or rate > 0.9 for rate in values)):
        raise argparse.ArgumentTypeError("missing rates must lie in [0.0, 0.9]")
    return values


def parse_modes(value):
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("at least one missing mode is required")
    invalid = [mode for mode in modes if mode not in VALID_MODES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported modes {invalid}; choose from {', '.join(VALID_MODES)}"
        )
    return list(dict.fromkeys(modes))


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def build_config(args):
    dataset = args.dataset.lower()
    video_cache_dir = (
        args.video_cache_dir
        or f"datasets/{('SIMS' if dataset == 'sims' else dataset.upper())}/video_cache/face32"
    )
    common = dict(
        dataset_name=dataset,
        num_hidden_layers=5,
        batch_size=args.batch_size,
        tasks="MTAV",
        seed=args.model_seed,
        trimodal_bottleneck_tokens=4,
        trimodal_av_cross=False,
        trimodal_tv_cross=True,
        trimodal_tv_cross_weight=0.2,
        ta_to_a_only=True,
        bidirectional_ta_cross=False,
        bidirectional_tv_cross=False,
        video_cache_dir=video_cache_dir,
        video_frames=32,
        video_model="clip_vitb32",
        video_chunk_size=args.video_chunk_size,
        video_local_only=True,
        video_temporal=True,
        video_temporal_layers=1,
        video_temporal_heads=4,
        video_temporal_dropout=0.1,
        video_pooling="mean",
        video_stage1_epochs=6,
        video_stage2_unfreeze_last_n=2,
        denoise=True,
        denoise_weight=0.1,
        denoise_sigma=0.2,
        denoise_tasks="MTAV",
        early_stop=getattr(args, "early_stop", 8),
    )
    if dataset == "sims":
        return ChConfig(
            learning_rate=args.lr if getattr(args, "lr", None) is not None else 1e-05,
            grad_accum_steps=1,
            **common,
        )
    return EnConfig(
        learning_rate=args.lr if getattr(args, "lr", None) is not None else 5e-06,
        context=True,
        text_context_len=2,
        audio_context_len=1,
        grad_accum_steps=4,
        video_lr_mult=1.0,
        v_corr_weight=0.3,
        v_cls_enable=True,
        v_cls_weight=0.5,
        v_cls_num_classes=7,
        v_cls_label_min=-3,
        v_cls_label_max=3,
        **common,
    )


def build_model(config, device):
    if config.dataset_name == "sims":
        model = ChineseModel(config)
    else:
        model = EnglishModel(config)
    return model.to(device)


def freeze_audio_frontend(model):
    for name in ("data2vec_model", "hubert_model"):
        backbone = getattr(model, name, None)
        feature_extractor = getattr(backbone, "feature_extractor", None)
        if feature_extractor is not None:
            for parameter in feature_extractor.parameters():
                parameter.requires_grad = False


def load_checkpoint(model, checkpoint, device):
    state = torch.load(checkpoint, map_location=device)
    metadata = state if isinstance(state, dict) else {}
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a PyTorch state_dict")
    if state and all((key.startswith("module.") for key in state)):
        state = {key[len("module.") :]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return metadata


def clone_batch(batch):
    cloned = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            cloned[key] = {
                subkey: subvalue.clone() if torch.is_tensor(subvalue) else subvalue
                for subkey, subvalue in value.items()
            }
        elif torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            moved[key] = {
                subkey: (
                    subvalue.to(device).view(-1, 1)
                    if torch.is_tensor(subvalue)
                    else subvalue
                )
                for subkey, subvalue in value.items()
            }
        elif torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    if torch.is_tensor(moved.get("targets")):
        moved["targets"] = moved["targets"].view(-1, 1)
    return moved


def stable_seed(mask_seed, sample_index, key):
    modulus = 2**63 - 1
    return (
        int(mask_seed) * 1000003 + int(sample_index) * 9176 + KEY_SEED_OFFSETS[key]
    ) % modulus


def row_rate(rate, row):
    if isinstance(rate, (list, tuple, np.ndarray)):
        return float(rate[row])
    if torch.is_tensor(rate) and rate.ndim > 0:
        return float(rate[row].item())
    return float(rate)


def corrupt_text(
    batch, key, mask_key, rate, mask_seed, sample_offset, special_ids, unk_id
):
    tokens = batch[key]
    attention = batch[mask_key]
    masked = 0
    eligible = 0
    special = torch.as_tensor(sorted(special_ids), dtype=tokens.dtype)
    for row in range(tokens.size(0)):
        valid = attention[row].bool()
        if special.numel() > 0:
            is_special = (tokens[row].unsqueeze(-1) == special).any(dim=-1)
            valid &= ~is_special
        positions = valid.nonzero(as_tuple=False).flatten()
        selected = select_bernoulli_subset(
            positions,
            row_rate(rate, row),
            stable_seed(mask_seed, sample_offset + row, key),
        )
        tokens[row, selected] = unk_id
        eligible += int(positions.numel())
        masked += int(selected.numel())
    return (masked, eligible)


def corrupt_audio(batch, key, mask_key, rate, mask_seed, sample_offset, block_size):
    audio = batch[key]
    attention = batch[mask_key]
    masked = 0
    eligible = 0
    for row in range(audio.size(0)):
        valid_length = int(attention[row].sum().item())
        block_count = (valid_length + block_size - 1) // block_size
        blocks = torch.arange(block_count, dtype=torch.long)
        selected = select_bernoulli_subset(
            blocks,
            row_rate(rate, row),
            stable_seed(mask_seed, sample_offset + row, key),
        )
        for block in selected.tolist():
            start = block * block_size
            end = min(valid_length, start + block_size)
            audio[row, start:end] = 0
        eligible += block_count
        masked += int(selected.numel())
    return (masked, eligible)


def corrupt_video(batch, key, mask_key, rate, mask_seed, sample_offset):
    frames = batch[key]
    frame_mask = batch[mask_key]
    masked = 0
    eligible = 0
    for row in range(frames.size(0)):
        positions = frame_mask[row].bool().nonzero(as_tuple=False).flatten()
        selected = select_bernoulli_subset(
            positions,
            row_rate(rate, row),
            stable_seed(mask_seed, sample_offset + row, key),
        )
        frames[row, selected] = 0
        frame_mask[row, selected] = 0
        eligible += int(positions.numel())
        masked += int(selected.numel())
    return (masked, eligible)


def count_audio_feature_units(batch, key, mask_key, rate, mask_seed, sample_offset):
    attention = batch[mask_key]
    masked = 0
    eligible = 0
    for row in range(attention.size(0)):
        units = (int(attention[row].sum().item()) + 319) // 320
        positions = torch.arange(units, dtype=torch.long)
        selected = select_bernoulli_subset(
            positions,
            row_rate(rate, row),
            stable_seed(mask_seed, sample_offset + row, key),
        )
        eligible += units
        masked += int(selected.numel())
    return (masked, eligible)


def count_video_feature_units(batch, key, mask_key, rate, mask_seed, sample_offset):
    frame_mask = batch[mask_key]
    masked = 0
    eligible = 0
    for row in range(frame_mask.size(0)):
        units = int(frame_mask[row].sum().item())
        positions = torch.arange(units, dtype=torch.long)
        selected = select_bernoulli_subset(
            positions,
            row_rate(rate, row),
            stable_seed(mask_seed, sample_offset + row, key),
        )
        eligible += units
        masked += int(selected.numel())
    return (masked, eligible)


def attach_feature_missing_spec(
    batch, audio_rates, video_rates, mask_seed, sample_offset
):
    batch_size = int(batch["text_tokens"].size(0))
    batch["_feature_missing_audio_rates"] = (
        torch.as_tensor(audio_rates, dtype=torch.float64).expand(batch_size).clone()
    )
    batch["_feature_missing_video_rates"] = (
        torch.as_tensor(video_rates, dtype=torch.float64).expand(batch_size).clone()
    )
    batch["_feature_missing_sample_indices"] = torch.arange(
        sample_offset, sample_offset + batch_size, dtype=torch.long
    )
    batch["_feature_missing_seed"] = int(mask_seed)


def set_model_feature_missing(model, batch):
    model._local_missing_spec = {
        "audio_rates": batch["_feature_missing_audio_rates"],
        "video_rates": batch["_feature_missing_video_rates"],
        "sample_indices": batch["_feature_missing_sample_indices"],
        "mask_seed": batch["_feature_missing_seed"],
    }


def clear_model_feature_missing(model):
    model._local_missing_spec = None


def apply_local_missing(
    batch, mode, rate, mask_seed, sample_offset, special_ids, unk_id, audio_block_size
):
    counts = {
        "masked_text_units": 0,
        "eligible_text_units": 0,
        "masked_audio_units": 0,
        "eligible_audio_units": 0,
        "masked_video_units": 0,
        "eligible_video_units": 0,
    }
    active = {"text", "audio", "video"} if mode == "all" or rate <= 0 else {mode}
    if "text" in active:
        for key, mask_key in (
            ("text_tokens", "text_masks"),
            ("text_context_tokens", "text_context_masks"),
        ):
            if key not in batch:
                continue
            masked, eligible = corrupt_text(
                batch,
                key,
                mask_key,
                rate,
                mask_seed,
                sample_offset,
                special_ids,
                unk_id,
            )
            counts["masked_text_units"] += masked
            counts["eligible_text_units"] += eligible
    if "audio" in active:
        for key, mask_key in (
            ("audio_inputs", "audio_masks"),
            ("audio_context_inputs", "audio_context_masks"),
        ):
            if key not in batch:
                continue
            masked, eligible = count_audio_feature_units(
                batch, key, mask_key, rate, mask_seed, sample_offset
            )
            counts["masked_audio_units"] += masked
            counts["eligible_audio_units"] += eligible
    if "video" in active:
        for key, mask_key in (
            ("video_inputs", "video_masks"),
            ("video_context_inputs", "video_context_masks"),
        ):
            if key not in batch:
                continue
            masked, eligible = count_video_feature_units(
                batch, key, mask_key, rate, mask_seed, sample_offset
            )
            counts["masked_video_units"] += masked
            counts["eligible_video_units"] += eligible
    batch_size = int(batch["text_tokens"].size(0))
    audio_rates = rate if "audio" in active else [0.0] * batch_size
    video_rates = rate if "video" in active else [0.0] * batch_size
    attach_feature_missing_spec(
        batch, audio_rates, video_rates, mask_seed, sample_offset
    )
    return counts


def sampled_nonclean_train_rate(train_seed, epoch, sample_index, modality):
    modality_offset = {"text": 101, "audio": 211, "video": 307}[modality]
    seed = (
        int(train_seed) * 1000003
        + int(epoch) * 100003
        + int(sample_index) * 9176
        + modality_offset
    ) % (2**63 - 1)
    rng = random.Random(seed)
    return rng.random()


def make_clean_position_sets(dataset_size, clean_probability, train_seed, epoch):
    clean_count = int(dataset_size * clean_probability)
    clean_sets = {}
    for modality, offset in (("text", 101), ("audio", 211), ("video", 307)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(train_seed + epoch * 100003 + offset)
        order = torch.randperm(dataset_size, generator=generator)
        clean_sets[modality] = set(order[:clean_count].tolist())
    return clean_sets


def apply_lnl_train_missing(
    batch,
    train_seed,
    epoch,
    sample_offset,
    special_ids,
    unk_id,
    audio_block_size,
    clean_probability,
    clean_position_sets=None,
):
    batch_size = int(batch["text_tokens"].size(0))
    if clean_position_sets is None:
        clean_position_sets = make_clean_position_sets(
            sample_offset + batch_size, clean_probability, train_seed, epoch
        )
    rates = {
        modality: [
            (
                0.0
                if sample_offset + row in clean_position_sets[modality]
                else sampled_nonclean_train_rate(
                    train_seed, epoch, sample_offset + row, modality
                )
            )
            for row in range(batch_size)
        ]
        for modality in ("text", "audio", "video")
    }
    mask_seed = train_seed + epoch * 100003
    counts = {
        "masked_text_units": 0,
        "eligible_text_units": 0,
        "masked_audio_units": 0,
        "eligible_audio_units": 0,
        "masked_video_units": 0,
        "eligible_video_units": 0,
    }
    for key, mask_key in (
        ("text_tokens", "text_masks"),
        ("text_context_tokens", "text_context_masks"),
    ):
        if key in batch:
            masked, eligible = corrupt_text(
                batch,
                key,
                mask_key,
                rates["text"],
                mask_seed,
                sample_offset,
                special_ids,
                unk_id,
            )
            counts["masked_text_units"] += masked
            counts["eligible_text_units"] += eligible
    for key, mask_key in (
        ("audio_inputs", "audio_masks"),
        ("audio_context_inputs", "audio_context_masks"),
    ):
        if key in batch:
            masked, eligible = count_audio_feature_units(
                batch, key, mask_key, rates["audio"], mask_seed, sample_offset
            )
            counts["masked_audio_units"] += masked
            counts["eligible_audio_units"] += eligible
    for key, mask_key in (
        ("video_inputs", "video_masks"),
        ("video_context_inputs", "video_context_masks"),
    ):
        if key in batch:
            masked, eligible = count_video_feature_units(
                batch, key, mask_key, rates["video"], mask_seed, sample_offset
            )
            counts["masked_video_units"] += masked
            counts["eligible_video_units"] += eligible
    attach_feature_missing_spec(
        batch, rates["audio"], rates["video"], mask_seed, sample_offset
    )
    return (counts, rates)


class LNLNTrainMissingLoader:

    def __init__(
        self,
        base_loader,
        epoch,
        train_seed,
        special_ids,
        unk_id,
        audio_block_size,
        clean_probability,
    ):
        self.base_loader = base_loader
        self.dataset = base_loader.dataset
        self.epoch = epoch
        self.train_seed = train_seed
        self.special_ids = special_ids
        self.unk_id = unk_id
        self.audio_block_size = audio_block_size
        self.clean_probability = clean_probability

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        sample_offset = 0
        clean_position_sets = make_clean_position_sets(
            len(self.dataset), self.clean_probability, self.train_seed, self.epoch
        )
        for original_batch in self.base_loader:
            batch = clone_batch(original_batch)
            apply_lnl_train_missing(
                batch=batch,
                train_seed=self.train_seed,
                epoch=self.epoch,
                sample_offset=sample_offset,
                special_ids=self.special_ids,
                unk_id=self.unk_id,
                audio_block_size=self.audio_block_size,
                clean_probability=self.clean_probability,
                clean_position_sets=clean_position_sets,
            )
            sample_offset += int(batch["text_tokens"].size(0))
            yield batch


class FixedMissingLoader:

    def __init__(
        self, base_loader, rate, mask_seed, special_ids, unk_id, audio_block_size
    ):
        self.base_loader = base_loader
        self.dataset = base_loader.dataset
        self.rate = rate
        self.mask_seed = mask_seed
        self.special_ids = special_ids
        self.unk_id = unk_id
        self.audio_block_size = audio_block_size

    def __len__(self):
        return len(self.base_loader)

    def __iter__(self):
        sample_offset = 0
        for original_batch in self.base_loader:
            batch = clone_batch(original_batch)
            apply_local_missing(
                batch=batch,
                mode="all",
                rate=self.rate,
                mask_seed=self.mask_seed,
                sample_offset=sample_offset,
                special_ids=self.special_ids,
                unk_id=self.unk_id,
                audio_block_size=self.audio_block_size,
            )
            sample_offset += int(batch["text_tokens"].size(0))
            yield batch


def batch_size_of(batch, dataset):
    if dataset == "sims":
        return int(batch["targets"]["M"].size(0))
    return int(batch["targets"].size(0))


def model_inputs(batch, dataset):
    if dataset == "sims":
        return (
            batch["text_tokens"],
            batch["text_masks"],
            batch["audio_inputs"],
            batch["audio_masks"],
            batch["video_inputs"],
            batch["video_masks"],
        )
    return (
        batch["text_tokens"],
        batch["text_masks"],
        batch["text_context_tokens"],
        batch["text_context_masks"],
        batch["audio_inputs"],
        batch["audio_masks"],
        batch["audio_context_inputs"],
        batch["audio_context_masks"],
        batch["video_inputs"],
        batch["video_masks"],
        batch["video_context_inputs"],
        batch["video_context_masks"],
    )


@torch.no_grad()
def evaluate_setting(
    model,
    loader,
    metrics,
    dataset,
    mode,
    rate,
    mask_seed,
    model_seed,
    special_ids,
    unk_id,
    audio_block_size,
    device,
):
    criterion = torch.nn.L1Loss(reduction="sum")
    predictions = []
    targets_all = []
    total_loss = 0.0
    total_items = 0
    totals = {
        "masked_text_units": 0,
        "eligible_text_units": 0,
        "masked_audio_units": 0,
        "eligible_audio_units": 0,
        "masked_video_units": 0,
        "eligible_video_units": 0,
    }
    model.eval()
    sample_offset = 0
    description = f"mode={mode} missing={rate:.1f}"
    for original_batch in tqdm(loader, desc=description):
        batch = clone_batch(original_batch)
        counts = apply_local_missing(
            batch=batch,
            mode=mode,
            rate=rate,
            mask_seed=mask_seed,
            sample_offset=sample_offset,
            special_ids=special_ids,
            unk_id=unk_id,
            audio_block_size=audio_block_size,
        )
        for key, value in counts.items():
            totals[key] += value
        current_batch_size = batch_size_of(batch, dataset)
        sample_offset += current_batch_size
        batch = move_batch(batch, device)
        targets = batch["targets"]["M"] if dataset == "sims" else batch["targets"]
        set_model_feature_missing(model, batch)
        try:
            outputs = model(*model_inputs(batch, dataset))
        finally:
            clear_model_feature_missing(model)
        prediction = outputs["M"]
        total_loss += criterion(prediction, targets).item()
        total_items += current_batch_size
        predictions.append(prediction.cpu())
        targets_all.append(targets.cpu())
    result = metrics(torch.cat(predictions), torch.cat(targets_all))
    result.update(totals)
    result.update(
        {
            "dataset": dataset,
            "mode": mode,
            "missing_rate": rate,
            "missing_percent": int(round(rate * 100)),
            "model_seed": model_seed,
            "mask_seed": mask_seed,
            "audio_block_size": audio_block_size,
            "total": total_items,
            "Loss": round(total_loss / max(1, total_items), 4),
        }
    )
    return result


def target_for_branch(batch, dataset, branch, device):
    if dataset == "sims":
        return batch["targets"][branch].to(device).view(-1, 1)
    return batch["targets"].to(device).view(-1, 1)


def build_training_optimizer(model, trainer, dataset, weight_decay):
    if dataset != "sims":
        optimizer = trainer._build_optimizer(model)
        for group in optimizer.param_groups:
            group["weight_decay"] = weight_decay
        return optimizer
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=trainer.config.learning_rate,
        weight_decay=weight_decay,
    )


def train_one_epoch(model, loader, trainer, optimizer, dataset, device):
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0
    total_items = 0
    accumulation_steps = max(1, int(trainer.config.grad_accum_steps))
    num_batches = len(loader)
    for step, original_batch in enumerate(tqdm(loader, desc="train"), start=1):
        batch = move_batch(original_batch, device)
        set_model_feature_missing(model, batch)
        try:
            outputs = model(*model_inputs(batch, dataset))
        finally:
            clear_model_feature_missing(model)
        loss = outputs["M"].new_tensor(0.0)
        for branch in trainer.loss_tasks:
            targets = target_for_branch(batch, dataset, branch, device)
            loss = loss + trainer.config.loss_weights[branch] * trainer.criterion(
                outputs[branch], targets
            )
        if dataset != "sims":
            main_targets = target_for_branch(batch, dataset, "M", device)
            if "A" in trainer.loss_tasks:
                loss = loss + trainer._branch_aux_loss(
                    outputs, main_targets, branch="A"
                )
            if "V" in trainer.loss_tasks:
                loss = loss + trainer._branch_aux_loss(
                    outputs, main_targets, branch="V"
                )
        if trainer.config.denoise:
            denoise_loss = outputs["M"].new_tensor(0.0)
            for branch in trainer.denoise_tasks:
                denoise_loss = denoise_loss + trainer.criterion(
                    outputs[f"{branch}_denoised"], outputs[f"{branch}_clean"]
                )
            loss = loss + trainer.config.denoise_weight * (
                denoise_loss / max(1, len(trainer.denoise_tasks))
            )
        batch_size = int(batch["text_tokens"].size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_items += batch_size
        (loss / accumulation_steps).backward()
        if step % accumulation_steps == 0 or step == num_batches:
            optimizer.step()
            optimizer.zero_grad()
    return total_loss / max(1, total_items)


def selection_spec(dataset):
    if dataset == "sims":
        return {
            "Mult_acc_2": "max",
            "Mult_acc_3": "max",
            "Mult_acc_5": "max",
            "MAE": "min",
        }
    return {
        "Has0_acc_2": "max",
        "Non0_acc_2": "max",
        "Mult_acc_5": "max",
        "Mult_acc_7": "max",
        "MAE": "min",
    }


def is_better(value, best_value, direction):
    if best_value is None:
        return True
    return value > best_value if direction == "max" else value < best_value


def save_training_checkpoint(path, model, optimizer, epoch, key, score, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "selection_key": key,
        "selection_score": score,
        "selection_split": args.selection_split,
        "selection_missing_rate": args.selection_rate,
        "model_seed": args.model_seed,
        "train_missing_policy": "lnln_half_clean_half_uniform",
        "training_input": "single randomly-corrupted view",
        "missing_protocol_version": "lnln_bernoulli_precontext_v3",
        "position_sampling": "independent Bernoulli per valid position",
        "feature_corruption_stage": "post-local-feature/pre-context-Transformer for audio and video",
        "state_dict": model.state_dict(),
    }
    if args.save_optimizer:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def checkpoint_path(save_dir, key, seed):
    return save_dir / f"best_{key}_{seed}.pth"


def output_fields(metric_keys):
    return [
        "dataset",
        "mode",
        "selection_key",
        "selection_protocol",
        "checkpoint_epoch",
        "missing_percent",
        "missing_rate",
        "model_seed",
        "mask_seed",
        *metric_keys,
        "masked_text_units",
        "eligible_text_units",
        "masked_audio_units",
        "eligible_audio_units",
        "masked_video_units",
        "eligible_video_units",
        "audio_block_size",
        "total",
    ]


def write_outputs(rows, csv_path, json_path, metric_keys):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = output_fields(metric_keys)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run_training(
    args, config, train_loader, test_loader, val_loader, tokenizer, device, metric_keys
):
    if args.selection_split == "test":
        print(
            "WARNING: --selection_split test reproduces LNLN's legacy official-code behavior but tunes checkpoints on the test set. Use validation selection for the primary paper result.",
            flush=True,
        )
    model = build_model(config, device)
    freeze_audio_frontend(model)
    if args.init_checkpoint:
        load_checkpoint(model, Path(args.init_checkpoint), device)
    trainer = ChTrainer(config) if args.dataset == "sims" else EnTrainer(config)
    optimizer = build_training_optimizer(
        model, trainer, args.dataset, args.weight_decay
    )
    metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
    special_ids = set(tokenizer.all_special_ids)
    save_dir = (
        Path(args.save_dir)
        if args.save_dir
        else Path("checkpoint/random_local_missing_bernoulli_precontext")
        / args.dataset
        / f"seed{args.model_seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    specs = selection_spec(args.dataset)
    best_scores = {key: None for key in specs}
    best_epochs = {key: None for key in specs}
    history = []
    best_selection_loss = None
    epochs_without_loss_improvement = 0
    selection_loader = test_loader if args.selection_split == "test" else val_loader
    for epoch in range(1, args.epochs + 1):
        stage_note = trainer.prepare_epoch(model, epoch - 1)
        if stage_note:
            print(f"epoch={epoch} {stage_note}")
        missing_train_loader = LNLNTrainMissingLoader(
            base_loader=train_loader,
            epoch=epoch,
            train_seed=args.train_missing_seed,
            special_ids=special_ids,
            unk_id=tokenizer.unk_token_id,
            audio_block_size=args.audio_block_size,
            clean_probability=args.train_clean_probability,
        )
        train_loss = train_one_epoch(
            model, missing_train_loader, trainer, optimizer, args.dataset, device
        )
        selection_results = evaluate_setting(
            model=model,
            loader=selection_loader,
            metrics=metrics,
            dataset=args.dataset,
            mode="all",
            rate=args.selection_rate,
            mask_seed=args.selection_mask_seed,
            model_seed=args.model_seed,
            special_ids=special_ids,
            unk_id=tokenizer.unk_token_id,
            audio_block_size=args.audio_block_size,
            device=device,
        )
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "selection_split": args.selection_split,
            "selection_rate": args.selection_rate,
            **{key: selection_results[key] for key in metric_keys},
        }
        history.append(history_row)
        (save_dir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} selection_MAE={selection_results['MAE']:.4f} selection_Corr={selection_results['Corr']:.4f}",
            flush=True,
        )
        for key, direction in specs.items():
            value = float(selection_results[key])
            if is_better(value, best_scores[key], direction):
                best_scores[key] = value
                best_epochs[key] = epoch
                save_training_checkpoint(
                    checkpoint_path(save_dir, key, args.model_seed),
                    model,
                    optimizer,
                    epoch,
                    key,
                    value,
                    args,
                )
        selection_loss = float(selection_results["Loss"])
        if best_selection_loss is None or selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            epochs_without_loss_improvement = 0
        else:
            epochs_without_loss_improvement += 1
        if args.early_stop > 0 and epochs_without_loss_improvement >= args.early_stop:
            print(
                f"Early stopping after {args.early_stop} epochs without selection Loss improvement."
            )
            break
    manifest = {
        "dataset": args.dataset,
        "model_seed": args.model_seed,
        "train_missing_seed": args.train_missing_seed,
        "train_clean_probability": args.train_clean_probability,
        "train_nonclean_rate_distribution": "Uniform(0,1)",
        "training_input": "single randomly-corrupted view",
        "missing_protocol_version": "lnln_bernoulli_precontext_v3",
        "position_sampling": "independent Bernoulli per valid position",
        "feature_corruption_stage": "post-local-feature/pre-context-Transformer for audio and video",
        "sims_train_test_duplicate_removed": args.dataset == "sims"
        and (not args.keep_sims_train_test_duplicate),
        "selection_split": args.selection_split,
        "selection_missing_rate": args.selection_rate,
        "selection_mask_seed": args.selection_mask_seed,
        "early_stop_criterion": "selection Loss",
        "best_selection_loss": best_selection_loss,
        "best_scores": best_scores,
        "best_epochs": best_epochs,
        "checkpoint_dir": str(save_dir.resolve()),
    }
    (save_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.skip_post_eval:
        return
    rows = []
    selection_protocol = f"lnln_bernoulli_precontext_v3:{args.selection_split}@missing_rate={args.selection_rate}"
    for selection_key in specs:
        path = checkpoint_path(save_dir, selection_key, args.model_seed)
        checkpoint_state = load_checkpoint(model, path, device)
        checkpoint_epoch = checkpoint_state.get("epoch", "")
        for rate in args.rates:
            row = evaluate_setting(
                model=model,
                loader=test_loader,
                metrics=metrics,
                dataset=args.dataset,
                mode="all",
                rate=rate,
                mask_seed=args.mask_seed,
                model_seed=args.model_seed,
                special_ids=special_ids,
                unk_id=tokenizer.unk_token_id,
                audio_block_size=args.audio_block_size,
                device=device,
            )
            row["selection_key"] = selection_key
            row["selection_protocol"] = selection_protocol
            row["checkpoint_epoch"] = checkpoint_epoch
            rows.append(row)
            print_result(row, args.dataset)
    result_base = save_dir / f"missing_eval_seed{args.model_seed}"
    write_outputs(
        rows,
        result_base.with_suffix(".csv"),
        result_base.with_suffix(".json"),
        metric_keys,
    )


def print_result(row, dataset):
    if dataset == "sims":
        score = f"Acc2={row['Mult_acc_2']:.4f} F1={row['F1_score']:.4f}"
    else:
        score = f"Acc2={row['Has0_acc_2']:.4f} F1={row['Has0_F1_score']:.4f}"
    print(
        f"mode={row['mode']:<5} rate={row['missing_percent']:>2}% {score} MAE={row['MAE']:.4f} Corr={row['Corr']:.4f} masked/eligible T={row['masked_text_units']}/{row['eligible_text_units']} A={row['masked_audio_units']}/{row['eligible_audio_units']} V={row['masked_video_units']}/{row['eligible_video_units']}",
        flush=True,
    )


def default_output_paths(args):
    checkpoint_stem = Path(args.checkpoint).stem
    base = (
        Path("runs/random_local_missing_bernoulli_precontext")
        / f"{args.dataset}_{checkpoint_stem}_maskseed{args.mask_seed}"
    )
    csv_path = Path(args.output_csv) if args.output_csv else base.with_suffix(".csv")
    json_path = (
        Path(args.output_json) if args.output_json else base.with_suffix(".json")
    )
    return (csv_path, json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train with LNLN-style random local missingness or evaluate a formal trimodal checkpoint under fixed missing rates."
    )
    parser.add_argument("--dataset", required=True, choices=("mosi", "mosei", "sims"))
    parser.add_argument(
        "--train",
        action="store_true",
        help="train from scratch with LNLN's half-clean/half-Uniform missing policy",
    )
    parser.add_argument("--checkpoint", help="checkpoint for evaluation-only mode")
    parser.add_argument(
        "--init_checkpoint", help="optional initialization checkpoint for --train"
    )
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=list(VALID_MODES),
        help="comma-separated: all,text,audio,video (default: all four)",
    )
    parser.add_argument(
        "--rates",
        type=parse_float_list,
        default=DEFAULT_RATES,
        help="comma-separated missing rates in [0,0.9]",
    )
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--mask_seed", type=int, default=20260825)
    parser.add_argument(
        "--train_missing_seed",
        type=int,
        help="training corruption seed (default: model_seed)",
    )
    parser.add_argument(
        "--train_clean_probability",
        type=float,
        default=0.5,
        help="per-sample, per-modality probability of no training corruption",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--early_stop",
        type=int,
        default=8,
        help="stop after this many epochs without selection MAE improvement; 0 disables",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--selection_split",
        choices=("valid", "test"),
        default="valid",
        help="checkpoint-selection split; test only reproduces LNLN's legacy code",
    )
    parser.add_argument(
        "--selection_rate",
        type=float,
        default=0.5,
        help="fixed all-modality missing rate used for checkpoint selection",
    )
    parser.add_argument("--selection_mask_seed", type=int, default=20260826)
    parser.add_argument("--save_dir")
    parser.add_argument(
        "--save_optimizer",
        action="store_true",
        help="also store optimizer state in every metric-specific checkpoint",
    )
    parser.add_argument("--skip_post_eval", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--audio_block_size",
        type=int,
        default=320,
        help="audio feature-stride metadata for reporting; the LNLN protocol zeros encoded audio time steps (default 320 samples at 16 kHz)",
    )
    parser.add_argument("--video_chunk_size", type=int, default=8)
    parser.add_argument("--video_cache_dir")
    parser.add_argument(
        "--keep_sims_train_test_duplicate",
        action="store_true",
        help="retain the known CH-SIMS train/test duplicate; by default the training copy is removed for leakage-safe robustness experiments",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--output_csv")
    parser.add_argument("--output_json")
    args = parser.parse_args()
    if args.audio_block_size <= 0:
        parser.error("--audio_block_size must be positive")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not 0.0 <= args.train_clean_probability <= 1.0:
        parser.error("--train_clean_probability must lie in [0,1]")
    if not 0.0 <= args.selection_rate <= 0.9:
        parser.error("--selection_rate must lie in [0,0.9]")
    if args.train_missing_seed is None:
        args.train_missing_seed = args.model_seed
    if args.train:
        if args.checkpoint:
            parser.error(
                "use --init_checkpoint, not --checkpoint, together with --train"
            )
        if args.init_checkpoint and (not Path(args.init_checkpoint).is_file()):
            parser.error(f"initial checkpoint does not exist: {args.init_checkpoint}")
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required unless --train is used")
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_file():
            parser.error(f"checkpoint does not exist: {checkpoint}")
    device = resolve_device(args.device)
    torch.manual_seed(args.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model_seed)
    np.random.seed(args.model_seed)
    random.seed(args.model_seed)
    torch.backends.cudnn.deterministic = True
    config = build_config(args)
    loader_kwargs = dict(
        batch_size=config.batch_size,
        dataset=config.dataset_name,
        video_cache_dir=config.video_cache_dir,
        video_frames=config.video_frames,
        drop_sims_train_test_duplicate=args.dataset == "sims"
        and (not args.keep_sims_train_test_duplicate),
    )
    if args.dataset != "sims":
        loader_kwargs.update(
            text_context_length=config.text_context_len,
            audio_context_length=config.audio_context_len,
        )
    train_loader, test_loader, val_loader = data_loader(**loader_kwargs)
    tokenizer = test_loader.dataset.tokenizer
    if tokenizer.unk_token_id is None:
        raise ValueError("the dataset tokenizer has no UNK token id")
    metric_keys = SIMS_METRICS if args.dataset == "sims" else ENGLISH_METRICS
    if args.train:
        run_training(
            args=args,
            config=config,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            metric_keys=metric_keys,
        )
        return
    special_ids = set(tokenizer.all_special_ids)
    model = build_model(config, device)
    freeze_audio_frontend(model)
    load_checkpoint(model, checkpoint, device)
    metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
    csv_path, json_path = default_output_paths(args)
    rows = []
    clean_row = None
    for mode in args.modes:
        for rate in args.rates:
            if rate <= 0 and clean_row is not None:
                row = dict(clean_row)
                row["mode"] = mode
            else:
                row = evaluate_setting(
                    model=model,
                    loader=test_loader,
                    metrics=metrics,
                    dataset=args.dataset,
                    mode=mode,
                    rate=rate,
                    mask_seed=args.mask_seed,
                    model_seed=args.model_seed,
                    special_ids=special_ids,
                    unk_id=tokenizer.unk_token_id,
                    audio_block_size=args.audio_block_size,
                    device=device,
                )
                if rate <= 0:
                    clean_row = dict(row)
            rows.append(row)
            write_outputs(rows, csv_path, json_path, metric_keys)
            print_result(row, args.dataset)
    print(f"\nSaved CSV: {csv_path.resolve()}")
    print(f"Saved JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
