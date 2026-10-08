import os
import torch
from torch import nn
from tqdm import tqdm
from utils.metrics import MetricsTop
from models.english import EnglishModel
from models.english_no_context import EnglishModelWithoutContext
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
    base_dir = config.model_save_path
    os.makedirs(base_dir, exist_ok=True)
    acc_path = os.path.join(base_dir, f"RH_acc_{config.dataset_name}_{config.seed}.pth")
    loss_path = os.path.join(
        base_dir, f"RH_loss_{config.dataset_name}_{config.seed}.pth"
    )
    return (acc_path, loss_path)


class EnConfig(object):
    """Configuration class to store the configurations of training."""

    def __init__(
        self,
        train_mode="regression",
        loss_weights={"M": 1, "T": 1, "A": 1, "V": 1},
        model_save_path="checkpoint/",
        learning_rate=1e-05,
        epochs=20,
        dataset_name="mosei",
        early_stop=8,
        seed=0,
        dropout=0.3,
        batch_size=16,
        multi_task=True,
        num_hidden_layers=1,
        tasks="M",
        context=True,
        text_context_len=2,
        audio_context_len=1,
        denoise=False,
        denoise_weight=0.1,
        denoise_sigma=0.2,
        denoise_tasks="MTA",
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
        grad_accum_steps=1,
        video_lr_mult=1.0,
        video_backbone_lr_mult=0.2,
        video_stage1_epochs=6,
        video_stage2_unfreeze_last_n=2,
        a_corr_weight=0.3,
        a_cls_enable=True,
        a_cls_weight=0.5,
        a_cls_num_classes=7,
        a_cls_label_min=-3.0,
        a_cls_label_max=3.0,
        v_corr_weight=0.3,
        v_cls_enable=True,
        v_cls_weight=0.5,
        v_cls_num_classes=7,
        v_cls_label_min=-3.0,
        v_cls_label_max=3.0,
        disable_unimodal_heads=False,
        m_only_supervision=False,
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
        self.multi_task = multi_task
        self.num_hidden_layers = num_hidden_layers
        self.tasks = tasks
        self.context = context
        self.text_context_len = text_context_len
        self.audio_context_len = audio_context_len
        self.denoise = denoise
        self.denoise_weight = denoise_weight
        self.denoise_sigma = denoise_sigma
        self.denoise_tasks = denoise_tasks
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
        self.grad_accum_steps = grad_accum_steps
        self.video_lr_mult = video_lr_mult
        self.video_backbone_lr_mult = video_backbone_lr_mult
        self.video_stage1_epochs = video_stage1_epochs
        self.video_stage2_unfreeze_last_n = video_stage2_unfreeze_last_n
        self.a_corr_weight = a_corr_weight
        self.a_cls_enable = a_cls_enable
        self.a_cls_weight = a_cls_weight
        self.a_cls_num_classes = a_cls_num_classes
        self.a_cls_label_min = a_cls_label_min
        self.a_cls_label_max = a_cls_label_max
        self.v_corr_weight = v_corr_weight
        self.v_cls_enable = v_cls_enable
        self.v_cls_weight = v_cls_weight
        self.v_cls_num_classes = v_cls_num_classes
        self.v_cls_label_min = v_cls_label_min
        self.v_cls_label_max = v_cls_label_max
        self.disable_unimodal_heads = disable_unimodal_heads
        self.m_only_supervision = m_only_supervision


class EnTrainer:

    def __init__(self, config):
        self.config = config
        self.criterion = (
            nn.L1Loss() if config.train_mode == "regression" else nn.CrossEntropyLoss()
        )
        self.cls_criterion = nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
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
        self.task_names = list(self.loss_tasks)

    def prepare_epoch(self, model, epoch_idx):
        if not hasattr(model, "video_encoder"):
            return None
        if int(self.config.video_stage1_epochs) <= 0:
            return None
        if epoch_idx < int(self.config.video_stage1_epochs):
            model.video_encoder.freeze_backbone()
            return "video_stage=1 freeze_backbone"
        last_n = int(self.config.video_stage2_unfreeze_last_n)
        if last_n <= 0:
            model.video_encoder.unfreeze_backbone_all()
            return "video_stage=2 unfreeze_backbone_all"
        actual = model.video_encoder.unfreeze_backbone_last_n(last_n)
        if actual < 0:
            return "video_stage=2 unfreeze_backbone_all"
        return f"video_stage=2 unfreeze_last_{actual}"

    def _build_optimizer(self, model):
        base_lr = self.config.learning_rate
        mult = float(self.config.video_lr_mult)
        bb_mult = float(self.config.video_backbone_lr_mult)
        if mult == 1.0 and bb_mult == 1.0:
            return torch.optim.AdamW(model.parameters(), lr=base_lr)
        video_param_ids = set()
        backbone_param_ids = set()
        if hasattr(model, "video_encoder"):
            video_param_ids.update(
                (id(p) for p in model.video_encoder.parameters() if p.requires_grad)
            )
            if hasattr(model.video_encoder, "backbone"):
                backbone_param_ids.update(
                    (
                        id(p)
                        for p in model.video_encoder.backbone.parameters()
                        if p.requires_grad
                    )
                )
        backbone_params = []
        video_params = []
        base_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in backbone_param_ids:
                backbone_params.append(p)
            elif pid in video_param_ids:
                video_params.append(p)
            else:
                base_params.append(p)
        param_groups = []
        if base_params:
            param_groups.append({"params": base_params, "lr": base_lr})
        if video_params:
            param_groups.append({"params": video_params, "lr": base_lr * mult})
        if backbone_params:
            param_groups.append(
                {"params": backbone_params, "lr": base_lr * mult * bb_mult}
            )
        if not param_groups:
            return torch.optim.AdamW(model.parameters(), lr=base_lr)
        return torch.optim.AdamW(param_groups)

    def _pearson_loss(self, preds, targets, eps=1e-08):
        preds = preds.view(-1)
        targets = targets.view(-1)
        if preds.numel() < 2:
            return preds.new_tensor(0.0)
        preds_centered = preds - preds.mean()
        targets_centered = targets - targets.mean()
        numerator = (preds_centered * targets_centered).sum()
        denominator = torch.sqrt(
            (preds_centered.pow(2).sum() + eps) * (targets_centered.pow(2).sum() + eps)
        )
        corr = numerator / denominator
        return 1.0 - corr

    def _regression_to_class(self, targets, branch="A"):
        labels = targets.view(-1)
        prefix = branch.lower()
        label_min = float(getattr(self.config, f"{prefix}_cls_label_min"))
        label_max = float(getattr(self.config, f"{prefix}_cls_label_max"))
        labels = torch.clamp(torch.round(labels), min=label_min, max=label_max)
        num_classes = int(getattr(self.config, f"{prefix}_cls_num_classes"))
        if num_classes == 7 and label_min == -3.0 and (label_max == 3.0):
            return (labels + 3.0).long()
        scaled = (labels - label_min) / max(1e-08, label_max - label_min)
        scaled = torch.clamp(
            (scaled * (num_classes - 1)).round(), min=0, max=num_classes - 1
        )
        return scaled.long()

    def _branch_aux_loss(self, outputs, targets, branch="A"):
        if branch not in outputs:
            return targets.new_tensor(0.0)
        aux_loss = targets.new_tensor(0.0)
        prefix = branch.lower()
        corr_weight = float(getattr(self.config, f"{prefix}_corr_weight"))
        if corr_weight > 0:
            aux_loss = aux_loss + corr_weight * self._pearson_loss(
                outputs[branch], targets
            )
        if (
            bool(getattr(self.config, f"{prefix}_cls_enable"))
            and float(getattr(self.config, f"{prefix}_cls_weight")) > 0
            and (f"{branch}_cls_logits" in outputs)
        ):
            cls_targets = self._regression_to_class(targets, branch=branch)
            aux_loss = aux_loss + float(
                getattr(self.config, f"{prefix}_cls_weight")
            ) * self.cls_criterion(outputs[f"{branch}_cls_logits"], cls_targets)
        return aux_loss

    def do_train(self, model, data_loader):
        model.train()
        optimizer = self._build_optimizer(model)
        total_loss = 0
        accum_steps = max(1, int(self.config.grad_accum_steps))
        num_batches = len(data_loader)
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(data_loader), start=1):
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            video_inputs = batch["video_inputs"].to(device)
            video_mask = batch["video_masks"].to(device)
            if self.config.context:
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)
                video_context_inputs = batch["video_context_inputs"].to(device)
                video_context_mask = batch["video_context_masks"].to(device)
            targets = batch["targets"].to(device).view(-1, 1)
            if self.config.context:
                outputs = model(
                    text_inputs,
                    text_mask,
                    text_context_inputs,
                    text_context_mask,
                    audio_inputs,
                    audio_mask,
                    audio_context_inputs,
                    audio_context_mask,
                    video_inputs,
                    video_mask,
                    video_context_inputs,
                    video_context_mask,
                )
            else:
                outputs = model(
                    text_inputs,
                    text_mask,
                    audio_inputs,
                    audio_mask,
                    video_inputs,
                    video_mask,
                )
            if self.config.multi_task:
                loss = 0.0
                for m in self.loss_tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(
                        outputs[m], targets
                    )
                    loss += sub_loss
                if "A" in self.loss_tasks:
                    loss += self._branch_aux_loss(outputs, targets, branch="A")
                if "V" in self.loss_tasks:
                    loss += self._branch_aux_loss(outputs, targets, branch="V")
                total_loss += loss.item() * text_inputs.size(0)
            else:
                loss = self.criterion(outputs["M"], targets)
                total_loss += loss.item() * text_inputs.size(0)
            if self.config.denoise:
                denoise_loss = 0.0
                for m in self.denoise_tasks:
                    denoise_loss += self.criterion(
                        outputs[f"{m}_denoised"], outputs[f"{m}_clean"]
                    )
                denoise_loss = denoise_loss / max(1, len(self.denoise_tasks))
                loss += self.config.denoise_weight * denoise_loss
            loss = loss / accum_steps
            loss.backward()
            if step % accum_steps == 0 or step == num_batches:
                optimizer.step()
                optimizer.zero_grad()
        total_loss = round(total_loss / len(data_loader.dataset), 4)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()
        if self.config.multi_task:
            y_pred = {task: [] for task in self.task_names}
            y_true = {task: [] for task in self.task_names}
            total_loss = 0
            val_loss = {task: 0 for task in self.task_names}
        else:
            y_pred = []
            y_true = []
            total_loss = 0
        with torch.no_grad():
            for batch in tqdm(data_loader):
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)
                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                video_inputs = batch["video_inputs"].to(device)
                video_mask = batch["video_masks"].to(device)
                if self.config.context:
                    audio_context_inputs = batch["audio_context_inputs"].to(device)
                    audio_context_mask = batch["audio_context_masks"].to(device)
                    video_context_inputs = batch["video_context_inputs"].to(device)
                    video_context_mask = batch["video_context_masks"].to(device)
                targets = batch["targets"].to(device).view(-1, 1)
                if self.config.context:
                    outputs = model(
                        text_inputs,
                        text_mask,
                        text_context_inputs,
                        text_context_mask,
                        audio_inputs,
                        audio_mask,
                        audio_context_inputs,
                        audio_context_mask,
                        video_inputs,
                        video_mask,
                        video_context_inputs,
                        video_context_mask,
                    )
                else:
                    outputs = model(
                        text_inputs,
                        text_mask,
                        audio_inputs,
                        audio_mask,
                        video_inputs,
                        video_mask,
                    )
                if self.config.multi_task:
                    loss = 0.0
                    for m in self.loss_tasks:
                        sub_loss = self.config.loss_weights[m] * self.criterion(
                            outputs[m], targets
                        )
                        loss += sub_loss
                        val_loss[m] += sub_loss.item() * text_inputs.size(0)
                    if "A" in self.loss_tasks:
                        loss += self._branch_aux_loss(outputs, targets, branch="A")
                    if "V" in self.loss_tasks:
                        loss += self._branch_aux_loss(outputs, targets, branch="V")
                    if self.config.denoise:
                        denoise_loss = 0.0
                        for m in self.denoise_tasks:
                            denoise_loss += self.criterion(
                                outputs[f"{m}_denoised"], outputs[f"{m}_clean"]
                            )
                        denoise_loss = denoise_loss / max(1, len(self.denoise_tasks))
                        loss += self.config.denoise_weight * denoise_loss
                    total_loss += loss.item() * text_inputs.size(0)
                    for m in self.loss_tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(targets.cpu())
                else:
                    loss = self.criterion(outputs["M"], targets)
                    if self.config.denoise:
                        denoise_loss = 0.0
                        for m in self.denoise_tasks:
                            denoise_loss += self.criterion(
                                outputs[f"{m}_denoised"], outputs[f"{m}_clean"]
                            )
                        denoise_loss = denoise_loss / max(1, len(self.denoise_tasks))
                        loss += self.config.denoise_weight * denoise_loss
                    total_loss += loss.item() * text_inputs.size(0)
                    y_pred.append(outputs["M"].cpu())
                    y_true.append(targets.cpu())
        if self.config.multi_task:
            for m in self.task_names:
                val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            loss_parts = [f"{task}_loss: {val_loss[task]}" for task in self.task_names]
            print(mode + " >> loss: ", total_loss, "  " + "  ".join(loss_parts))
            eval_results = {}
            for m in self.task_names:
                pred, true = (torch.cat(y_pred[m]), torch.cat(y_true[m]))
                results = self.metrics(pred, true)
                print("%s: >> " % m + dict_to_str(results))
                eval_results[m] = results
            eval_results = eval_results[self.task_names[0]]
            eval_results["Loss"] = total_loss
        else:
            total_loss = round(total_loss / len(data_loader.dataset), 4)
            print(mode + " >> loss: ", total_loss)
            pred, true = (torch.cat(y_pred), torch.cat(y_true))
            eval_results = self.metrics(pred, true)
            print("%s: >> " % "M" + dict_to_str(eval_results))
            eval_results["Loss"] = total_loss
        return eval_results


def EnRun(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    if config.context:
        print(
            f"Trimodal mode: enabling shared audio/video context with context_len={config.audio_context_len}."
        )
    elif config.context:
        print(
            f"Video mode: enabling video context with context_len={config.audio_context_len}."
        )
    train_loader, test_loader, val_loader = data_loader(
        config.batch_size,
        config.dataset_name,
        text_context_length=config.text_context_len,
        audio_context_length=config.audio_context_len,
        video_cache_dir=config.video_cache_dir,
        video_frames=config.video_frames,
    )
    if config.context:
        model = EnglishModel(config).to(device)
        if hasattr(model, "data2vec_model"):
            for param in model.data2vec_model.feature_extractor.parameters():
                param.requires_grad = False
    else:
        model = EnglishModelWithoutContext(config).to(device)
        if hasattr(model, "data2vec_model"):
            for param in model.data2vec_model.feature_extractor.parameters():
                param.requires_grad = False
    trainer = EnTrainer(config)
    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    acc_ckpt_path, loss_ckpt_path = _checkpoint_paths(config)
    if any((os.path.exists(path) for path in (acc_ckpt_path, loss_ckpt_path))):
        raise FileExistsError("Checkpoint already exists; use a new model_save_path.")
    while True:
        print("---------------------EPOCH: ", epoch, "--------------------")
        stage_info = trainer.prepare_epoch(model, epoch)
        if stage_info:
            print(stage_info)
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")
        if eval_results["Loss"] < lowest_eval_loss:
            lowest_eval_loss = eval_results["Loss"]
            torch.save(model.state_dict(), loss_ckpt_path)
            best_epoch = epoch
        if eval_results["Has0_acc_2"] >= highest_eval_acc:
            highest_eval_acc = eval_results["Has0_acc_2"]
            torch.save(model.state_dict(), acc_ckpt_path)
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(acc_ckpt_path))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print("%s: >> " % "TEST (highest val acc) " + dict_to_str(test_results_loss))
    model.load_state_dict(torch.load(loss_ckpt_path))
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print("%s: >> " % "TEST (lowest val loss) " + dict_to_str(test_results_acc))
