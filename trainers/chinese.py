import os
import torch
from torch import nn
from tqdm import tqdm
from utils.metrics import MetricsTop
from models.chinese import ChineseModel
import random
import numpy as np
from data.dataset import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str


def _checkpoint_paths(config):
    """Return seed-specific paths so formal SIMS runs do not overwrite each other."""
    stem = f"{config.dataset_name}_{config.seed}"
    return (
        os.path.join(config.model_save_path, f"CH_acc_{stem}.pth"),
        os.path.join(config.model_save_path, f"CH_loss_{stem}.pth"),
    )


class ChConfig(object):
    """Configuration class to store the configurations of training."""

    def __init__(
        self,
        train_mode="regression",
        loss_weights={"M": 1, "T": 1, "A": 1, "V": 1},
        model_save_path="checkpoint/",
        learning_rate=1e-05,
        epochs=20,
        dataset_name="sims",
        early_stop=8,
        seed=0,
        dropout=0.3,
        batch_size=64,
        tasks="MTA",
        num_hidden_layers=3,
        denoise=False,
        denoise_weight=0.1,
        denoise_sigma=0.2,
        denoise_tasks="MTA",
        grad_accum_steps=1,
        video_cache_dir=None,
        video_frames=32,
        video_model="clip_vitb32",
        video_chunk_size=0,
        video_local_only=True,
        video_temporal=False,
        video_temporal_layers=1,
        video_temporal_heads=4,
        video_temporal_dropout=0.1,
        video_temporal_ffn_mult=4.0,
        video_pooling="mean",
        trimodal_bottleneck_tokens=4,
        trimodal_av_cross=False,
        trimodal_av_cross_weight=0.2,
        trimodal_tv_cross=False,
        trimodal_tv_cross_weight=0.2,
        disable_bottleneck=False,
        disable_ta_cross=False,
        ta_to_a_only=False,
        bidirectional_ta_cross=False,
        bidirectional_tv_cross=False,
        disable_unimodal_heads=False,
        m_only_supervision=False,
        video_stage1_epochs=6,
        video_stage2_unfreeze_last_n=2,
    ):
        self.train_mode = train_mode
        self.loss_weights = loss_weights
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.model_save_path = model_save_path
        self.early_stop = early_stop
        self.seed = seed
        self.dropout = dropout
        self.batch_size = batch_size
        self.tasks = tasks
        self.num_hidden_layers = num_hidden_layers
        self.denoise = denoise
        self.denoise_weight = denoise_weight
        self.denoise_sigma = denoise_sigma
        self.denoise_tasks = denoise_tasks
        self.grad_accum_steps = grad_accum_steps
        self.video_cache_dir = video_cache_dir
        self.video_frames = video_frames
        self.video_model = video_model
        self.video_chunk_size = video_chunk_size
        self.video_local_only = video_local_only
        self.video_temporal = video_temporal
        self.video_temporal_layers = video_temporal_layers
        self.video_temporal_heads = video_temporal_heads
        self.video_temporal_dropout = video_temporal_dropout
        self.video_temporal_ffn_mult = video_temporal_ffn_mult
        self.video_pooling = video_pooling
        self.trimodal_bottleneck_tokens = trimodal_bottleneck_tokens
        self.trimodal_av_cross = trimodal_av_cross
        self.trimodal_av_cross_weight = trimodal_av_cross_weight
        self.trimodal_tv_cross = trimodal_tv_cross
        self.trimodal_tv_cross_weight = trimodal_tv_cross_weight
        self.disable_bottleneck = disable_bottleneck
        self.disable_ta_cross = disable_ta_cross
        self.ta_to_a_only = ta_to_a_only
        self.bidirectional_ta_cross = bidirectional_ta_cross
        self.bidirectional_tv_cross = bidirectional_tv_cross
        self.disable_unimodal_heads = disable_unimodal_heads
        self.m_only_supervision = m_only_supervision
        self.video_stage1_epochs = video_stage1_epochs
        self.video_stage2_unfreeze_last_n = video_stage2_unfreeze_last_n


class ChTrainer:

    def __init__(self, config):
        self.config = config
        self.tasks = config.tasks
        self.loss_tasks = (
            "M"
            if getattr(config, "m_only_supervision", False)
            or getattr(config, "disable_unimodal_heads", False)
            else config.tasks
        )
        self.denoise_tasks = (
            "M"
            if getattr(config, "disable_unimodal_heads", False)
            else config.denoise_tasks
        )
        self.criterion = (
            nn.L1Loss() if config.train_mode == "regression" else nn.CrossEntropyLoss()
        )
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.task_names = list(self.loss_tasks)

    def prepare_epoch(self, model, epoch_idx):
        if not hasattr(model, "video_encoder"):
            return None
        if int(getattr(self.config, "video_stage1_epochs", 0)) <= 0:
            return None
        if epoch_idx < int(self.config.video_stage1_epochs):
            model.video_encoder.freeze_backbone()
            return "video_stage=1 freeze_backbone"
        last_n = int(getattr(self.config, "video_stage2_unfreeze_last_n", 0))
        if last_n <= 0:
            model.video_encoder.unfreeze_backbone_all()
            return "video_stage=2 unfreeze_backbone_all"
        actual = model.video_encoder.unfreeze_backbone_last_n(last_n)
        if actual < 0:
            return "video_stage=2 unfreeze_backbone_all"
        return f"video_stage=2 unfreeze_last_{actual}"

    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        total_loss = 0
        input_size = 0
        accum_steps = max(1, int(self.config.grad_accum_steps))
        num_batches = len(data_loader)
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(data_loader), start=1):
            text_inputs = batch["text_tokens"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            video_inputs = batch["video_inputs"].to(device)
            video_mask = batch["video_masks"].to(device)
            targets = batch["targets"]
            outputs = model(
                text_inputs,
                text_mask,
                audio_inputs,
                audio_mask,
                video_inputs,
                video_mask,
            )
            loss = 0.0
            for m in self.loss_tasks:
                sub_loss = self.config.loss_weights[m] * self.criterion(
                    outputs[m], targets[m].to(device).view(-1, 1)
                )
                loss += sub_loss
            if self.config.denoise:
                denoise_loss = 0.0
                for m in self.denoise_tasks:
                    denoise_loss += self.criterion(
                        outputs[f"{m}_denoised"], outputs[f"{m}_clean"]
                    )
                denoise_loss = denoise_loss / max(1, len(self.denoise_tasks))
                loss += self.config.denoise_weight * denoise_loss
            total_loss += loss.item() * text_inputs.size(0)
            input_size += text_inputs.size(0)
            loss = loss / accum_steps
            loss.backward()
            if step % accum_steps == 0 or step == num_batches:
                optimizer.step()
                optimizer.zero_grad()
        total_loss = round(total_loss / input_size, 4)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()
        y_pred = {m: [] for m in self.task_names}
        y_true = {m: [] for m in self.task_names}
        total_loss = 0
        val_loss = {m: 0 for m in self.task_names}
        input_size = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):
                text_inputs = batch["text_tokens"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                text_mask = batch["text_masks"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                video_inputs = batch["video_inputs"].to(device)
                video_mask = batch["video_masks"].to(device)
                targets = batch["targets"]
                outputs = model(
                    text_inputs,
                    text_mask,
                    audio_inputs,
                    audio_mask,
                    video_inputs,
                    video_mask,
                )
                loss = 0.0
                for m in self.loss_tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(
                        outputs[m], targets[m].to(device).view(-1, 1)
                    )
                    loss += sub_loss
                    val_loss[m] += sub_loss.item() * text_inputs.size(0)
                if self.config.denoise:
                    denoise_loss = 0.0
                    for m in self.denoise_tasks:
                        denoise_loss += self.criterion(
                            outputs[f"{m}_denoised"], outputs[f"{m}_clean"]
                        )
                    denoise_loss = denoise_loss / max(1, len(self.denoise_tasks))
                    loss += self.config.denoise_weight * denoise_loss
                total_loss += loss.item() * text_inputs.size(0)
                input_size += text_inputs.size(0)
                for m in self.loss_tasks:
                    y_pred[m].append(outputs[m].cpu())
                    y_true[m].append(targets[m].cpu())
        for m in self.loss_tasks:
            val_loss[m] = round(val_loss[m] / input_size, 4)
        total_loss = round(total_loss / input_size, 4)
        metric_line = "  ".join((f"{m}_loss: {val_loss[m]}" for m in self.task_names))
        print(mode + " >> loss: ", total_loss, "  ", metric_line)
        eval_results = {}
        for m in self.loss_tasks:
            pred, true = (torch.cat(y_pred[m]), torch.cat(y_true[m]))
            results = self.metrics(pred, true)
            print("%s: >> " % m + dict_to_str(results))
            eval_results[m] = results
        eval_results = eval_results[self.tasks[0]]
        eval_results["Loss"] = total_loss
        return eval_results


def ChRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    os.makedirs(config.model_save_path, exist_ok=True)
    train_loader, test_loader, val_loader = data_loader(
        config.batch_size,
        config.dataset_name,
        video_cache_dir=config.video_cache_dir,
        video_frames=config.video_frames,
    )
    model = ChineseModel(config).to(device)
    for param in model.hubert_model.feature_extractor.parameters():
        param.requires_grad = False
    trainer = ChTrainer(config)
    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    acc_ckpt_path, loss_ckpt_path = _checkpoint_paths(config)
    if any((os.path.exists(path) for path in (acc_ckpt_path, loss_ckpt_path))):
        raise FileExistsError("Checkpoint already exists; use a new model_save_path.")
    while epoch < config.epochs:
        print("---------------------EPOCH: ", epoch, "--------------------")
        stage_note = trainer.prepare_epoch(model, epoch)
        if stage_note:
            print(stage_note)
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")
        if eval_results["Loss"] < lowest_eval_loss:
            lowest_eval_loss = eval_results["Loss"]
            torch.save(model.state_dict(), loss_ckpt_path)
            best_epoch = epoch
        if eval_results["Mult_acc_2"] >= highest_eval_acc:
            highest_eval_acc = eval_results["Mult_acc_2"]
            torch.save(model.state_dict(), acc_ckpt_path)
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(acc_ckpt_path))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print("%s: >> " % "TEST (highest val acc) " + dict_to_str(test_results_loss))
    model.load_state_dict(torch.load(loss_ckpt_path))
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print("%s: >> " % "TEST (lowest val loss) " + dict_to_str(test_results_acc))
