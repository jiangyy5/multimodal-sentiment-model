import os
import torch
from torch import nn
from models.missing import apply_feature_missing
from transformers import AutoImageProcessor, CLIPVisionModel
from models.paths import resolve_model


def _get_encoder_layers(model):
    return model.vision_model.encoder.layers


class VideoEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.video_model not in ("clip_vitb32", "openai/clip-vit-base-patch32"):
            raise ValueError("Only OpenAI CLIP ViT-B/32 is supported")
        model_id = resolve_model(
            "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch32"
        )
        local_only = getattr(config, "video_local_only", True) or os.path.isdir(
            model_id
        )
        self.backbone = CLIPVisionModel.from_pretrained(
            model_id, local_files_only=local_only
        )
        self.hidden_size = self.backbone.config.hidden_size
        self.out_dim = getattr(config, "video_out_dim", 768)
        self.proj = None
        if self.hidden_size != self.out_dim:
            self.proj = nn.Linear(self.hidden_size, self.out_dim)
        self.pooling = str(getattr(config, "video_pooling", "mean")).lower()
        if self.pooling not in {"mean", "cls", "attn"}:
            self.pooling = "mean"
        self.attn_pool = None
        if self.pooling == "attn":
            self.attn_pool = nn.Sequential(
                nn.Linear(self.out_dim, self.out_dim),
                nn.Tanh(),
                nn.Linear(self.out_dim, 1),
            )
        self.temporal_enabled = getattr(config, "video_temporal", False)
        if self.temporal_enabled:
            layers = int(getattr(config, "video_temporal_layers", 1))
            heads = int(getattr(config, "video_temporal_heads", 4))
            dropout = float(getattr(config, "video_temporal_dropout", 0.1))
            ffn_mult = float(getattr(config, "video_temporal_ffn_mult", 4.0))
            dim_ffn = int(self.out_dim * ffn_mult)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.out_dim,
                nhead=heads,
                dim_feedforward=dim_ffn,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.temporal_cls = nn.Parameter(torch.zeros(1, 1, self.out_dim))
            max_len = int(getattr(config, "video_frames", 32)) + 1
            self.temporal_pos_emb = nn.Parameter(torch.zeros(1, max_len, self.out_dim))
            nn.init.normal_(self.temporal_cls, std=0.02)
            nn.init.normal_(self.temporal_pos_emb, std=0.02)
        self.chunk_size = getattr(config, "video_chunk_size", 0)
        try:
            processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=local_only
            )
            mean = torch.tensor(processor.image_mean).view(1, 1, 3, 1, 1)
            std = torch.tensor(processor.image_std).view(1, 1, 3, 1, 1)
        except Exception:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_backbone_last_n(self, last_n: int) -> int:
        layers = _get_encoder_layers(self.backbone)
        if layers is None:
            self.unfreeze_backbone_all()
            return -1
        self.freeze_backbone()
        total_layers = len(layers)
        if total_layers == 0:
            return 0
        keep_n = max(1, min(int(last_n), total_layers))
        for layer in layers[-keep_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for name, p in self.backbone.named_parameters():
            if (
                ".layernorm" in name
                or name.endswith(".layernorm.weight")
                or name.endswith(".layernorm.bias")
            ):
                p.requires_grad = True
            if ".pooler." in name:
                p.requires_grad = True
        return keep_n

    def _mask_to_float(self, mask, seq):
        if mask is None:
            return torch.ones(
                seq.size(0), seq.size(1), device=seq.device, dtype=torch.float32
            )
        return mask.to(device=seq.device, dtype=torch.float32)

    def _masked_mean_pool(self, seq, mask):
        mask = self._mask_to_float(mask, seq)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (seq * mask.unsqueeze(-1)).sum(dim=1) / denom
        return (pooled, mask)

    def _masked_attn_pool(self, seq, mask):
        mask = self._mask_to_float(mask, seq)
        logits = self.attn_pool(seq).squeeze(-1)
        logits = logits.masked_fill(mask <= 0, -10000.0)
        weights = torch.softmax(logits, dim=1)
        pooled = (seq * weights.unsqueeze(-1)).sum(dim=1)
        return (pooled, mask)

    def _pool_sequence(self, seq, mask):
        if self.pooling == "attn" and self.attn_pool is not None:
            return self._masked_attn_pool(seq, mask)
        if self.pooling == "cls":
            mask = self._mask_to_float(mask, seq)
            return (seq[:, 0, :], mask)
        return self._masked_mean_pool(seq, mask)

    def _normalize(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dtype != torch.float32:
            frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0
        return (frames - self.pixel_mean) / self.pixel_std

    def forward(
        self,
        video_inputs: torch.Tensor,
        video_mask: torch.Tensor = None,
        missing_owner=None,
        missing_key: str = "video",
    ):
        b, t, c, h, w = video_inputs.shape
        x = self._normalize(video_inputs)
        x = x.view(b * t, c, h, w)
        if self.chunk_size and self.chunk_size < x.size(0):
            pooled = []
            for start in range(0, x.size(0), self.chunk_size):
                out = self.backbone(
                    pixel_values=x[start : start + self.chunk_size], return_dict=True
                )
                pooled.append(
                    out.pooler_output
                    if out.pooler_output is not None
                    else out.last_hidden_state[:, 0, :]
                )
            frame_features = torch.cat(pooled, dim=0)
        else:
            out = self.backbone(pixel_values=x, return_dict=True)
            frame_features = (
                out.pooler_output
                if out.pooler_output is not None
                else out.last_hidden_state[:, 0, :]
            )
        seq_features = frame_features.view(b, t, -1)
        if self.proj is not None:
            seq_features = self.proj(seq_features)
        if missing_owner is not None and getattr(
            missing_owner, "_local_missing_spec", None
        ):
            seq_features = apply_feature_missing(
                missing_owner, seq_features, video_mask, "video", missing_key
            )
        if self.temporal_enabled:
            cls_token = self.temporal_cls.expand(b, 1, -1)
            x = torch.cat([cls_token, seq_features], dim=1)
            if x.size(1) <= self.temporal_pos_emb.size(1):
                pos = self.temporal_pos_emb[:, : x.size(1), :]
            else:
                extra = x.size(1) - self.temporal_pos_emb.size(1)
                last = self.temporal_pos_emb[:, -1:, :].expand(1, extra, -1)
                pos = torch.cat([self.temporal_pos_emb, last], dim=1)
            x = x + pos
            key_padding_mask = None
            if video_mask is not None:
                mask = ~video_mask.to(dtype=torch.bool, device=x.device)
                cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([cls_mask, mask], dim=1)
            x = self.temporal_encoder(x, src_key_padding_mask=key_padding_mask)
            seq_features = x[:, 1:, :]
            if self.pooling == "cls":
                pooled_features = x[:, 0, :]
                video_mask = self._mask_to_float(video_mask, seq_features)
            else:
                pooled_features, video_mask = self._pool_sequence(
                    seq_features, video_mask
                )
            return (seq_features, pooled_features, video_mask)
        pooled_features, video_mask = self._pool_sequence(seq_features, video_mask)
        return (seq_features, pooled_features, video_mask)
