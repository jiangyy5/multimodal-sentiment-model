import torch
from torch import nn


def _mask_to_bool(mask, seq):
    if mask is None:
        return torch.ones(seq.size(0), seq.size(1), device=seq.device, dtype=torch.bool)
    return mask.to(device=seq.device, dtype=torch.bool)


def _masked_mean(seq, mask):
    mask = _mask_to_bool(mask, seq)
    weights = mask.to(dtype=seq.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (seq * weights).sum(dim=1) / denom


def _stabilize_attention_mask(mask):
    mask = mask.clone()
    empty_rows = mask.sum(dim=1) == 0
    if empty_rows.any():
        mask[empty_rows, 0] = True
    return mask


class CrossResidual(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key_value, key_value_mask=None):
        key_padding_mask = None
        if key_value_mask is not None:
            key_padding_mask = ~_mask_to_bool(key_value_mask, key_value)
        attn_output, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.norm(query + self.dropout(attn_output))


class FeedForwardResidual(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        inner = hidden_size * 4
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.norm(x + self.dropout(self.net(x)))


class TriModalFusionBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout,
        av_cross=False,
        av_cross_weight=0.2,
        tv_cross=False,
        tv_cross_weight=0.2,
        disable_bottleneck=False,
        disable_ta_cross=False,
        ta_to_a_only=False,
        bidirectional_ta_cross=False,
        bidirectional_tv_cross=False,
    ):
        super().__init__()
        self.av_cross = av_cross
        self.av_cross_weight = av_cross_weight
        self.tv_cross = tv_cross
        self.tv_cross_weight = tv_cross_weight
        self.disable_bottleneck = disable_bottleneck
        self.disable_ta_cross = disable_ta_cross
        self.ta_to_a_only = ta_to_a_only
        self.bidirectional_ta_cross = bidirectional_ta_cross
        self.bidirectional_tv_cross = bidirectional_tv_cross
        self.text_self = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.audio_self = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.video_self = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.z_from_all = CrossResidual(hidden_size, num_heads, dropout)
        self.text_from_z = CrossResidual(hidden_size, num_heads, dropout)
        self.audio_from_z = CrossResidual(hidden_size, num_heads, dropout)
        self.video_from_z = CrossResidual(hidden_size, num_heads, dropout)
        self.text_from_audio = CrossResidual(hidden_size, num_heads, dropout)
        self.audio_from_text = CrossResidual(hidden_size, num_heads, dropout)
        self.audio_from_video = (
            CrossResidual(hidden_size, num_heads, dropout) if av_cross else None
        )
        self.video_from_audio = (
            CrossResidual(hidden_size, num_heads, dropout) if av_cross else None
        )
        self.video_from_text = (
            CrossResidual(hidden_size, num_heads, dropout) if tv_cross else None
        )
        self.text_from_video = (
            CrossResidual(hidden_size, num_heads, dropout)
            if tv_cross and bidirectional_tv_cross
            else None
        )
        self.text_ffn = FeedForwardResidual(hidden_size, dropout)
        self.audio_ffn = FeedForwardResidual(hidden_size, dropout)
        self.video_ffn = FeedForwardResidual(hidden_size, dropout)
        self.z_ffn = FeedForwardResidual(hidden_size, dropout)

    def forward(
        self, text, text_mask, audio, audio_mask, video, video_mask, bottleneck
    ):
        text_mask = _mask_to_bool(text_mask, text)
        audio_mask = _mask_to_bool(audio_mask, audio)
        video_mask = _mask_to_bool(video_mask, video)

        text_attn_mask = _stabilize_attention_mask(text_mask)
        audio_attn_mask = _stabilize_attention_mask(audio_mask)
        video_attn_mask = _stabilize_attention_mask(video_mask)

        text_pad = ~text_attn_mask
        audio_pad = ~audio_attn_mask
        video_pad = ~video_attn_mask

        text = self.text_self(text, src_key_padding_mask=text_pad)
        audio = self.audio_self(audio, src_key_padding_mask=audio_pad)
        video = self.video_self(video, src_key_padding_mask=video_pad)

        joint = torch.cat((text, audio, video), dim=1)
        joint_mask = torch.cat(
            (
                text_attn_mask,
                audio_attn_mask,
                video_attn_mask,
            ),
            dim=1,
        )
        if not self.disable_bottleneck:
            bottleneck = self.z_from_all(bottleneck, joint, joint_mask)
            bottleneck = self.z_ffn(bottleneck)

            text = self.text_from_z(text, bottleneck)
            audio = self.audio_from_z(audio, bottleneck)
            video = self.video_from_z(video, bottleneck)

        if not self.disable_ta_cross:
            if (not self.ta_to_a_only) or self.bidirectional_ta_cross:
                text = self.text_from_audio(text, audio, audio_attn_mask)
            audio = self.audio_from_text(audio, text, text_attn_mask)
        if self.av_cross:
            av_weight = float(self.av_cross_weight)
            audio_av = self.audio_from_video(audio, video, video_attn_mask)
            video_av = self.video_from_audio(video, audio, audio_attn_mask)
            audio = audio + av_weight * (audio_av - audio)
            video = video + av_weight * (video_av - video)
        if self.tv_cross:
            tv_weight = float(self.tv_cross_weight)
            video_tv = self.video_from_text(video, text, text_attn_mask)
            text_tv = None
            if self.bidirectional_tv_cross:
                text_tv = self.text_from_video(text, video, video_attn_mask)
            video = video + tv_weight * (video_tv - video)
            if text_tv is not None:
                text = text + tv_weight * (text_tv - text)

        text = self.text_ffn(text)
        audio = self.audio_ffn(audio)
        video = self.video_ffn(video)
        text = text * text_mask.to(dtype=text.dtype).unsqueeze(-1)
        audio = audio * audio_mask.to(dtype=audio.dtype).unsqueeze(-1)
        video = video * video_mask.to(dtype=video.dtype).unsqueeze(-1)
        return text, audio, video, bottleneck


class TriModalFusionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_layers=2,
        num_heads=12,
        bottleneck_tokens=4,
        dropout=0.3,
        av_cross=False,
        av_cross_weight=0.2,
        tv_cross=False,
        tv_cross_weight=0.2,
        disable_bottleneck=False,
        disable_ta_cross=False,
        ta_to_a_only=False,
        bidirectional_ta_cross=False,
        bidirectional_tv_cross=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_tokens = bottleneck_tokens
        self.disable_bottleneck = disable_bottleneck
        self.out_dim = hidden_size * (3 if disable_bottleneck else 4)
        self.bottleneck = None
        if not disable_bottleneck:
            self.bottleneck = nn.Parameter(
                torch.randn(1, bottleneck_tokens, hidden_size) * 0.02
            )
        self.layers = nn.ModuleList(
            [
                TriModalFusionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    av_cross=av_cross,
                    av_cross_weight=av_cross_weight,
                    tv_cross=tv_cross,
                    tv_cross_weight=tv_cross_weight,
                    disable_bottleneck=disable_bottleneck,
                    disable_ta_cross=disable_ta_cross,
                    ta_to_a_only=ta_to_a_only,
                    bidirectional_ta_cross=bidirectional_ta_cross,
                    bidirectional_tv_cross=bidirectional_tv_cross,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, text, text_mask, audio, audio_mask, video, video_mask):
        bottleneck = None
        if self.bottleneck is not None:
            bottleneck = self.bottleneck.expand(text.size(0), -1, -1)
        for layer in self.layers:
            text, audio, video, bottleneck = layer(
                text,
                text_mask,
                audio,
                audio_mask,
                video,
                video_mask,
                bottleneck,
            )

        text_pool = text[:, 0, :]
        audio_pool = _masked_mean(audio, audio_mask)
        video_pool = _masked_mean(video, video_mask)
        if self.disable_bottleneck:
            return torch.cat((text_pool, audio_pool, video_pool), dim=1)
        bottleneck_pool = bottleneck.mean(dim=1)
        return torch.cat((text_pool, audio_pool, video_pool, bottleneck_pool), dim=1)
