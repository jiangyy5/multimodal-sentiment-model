import os
import torch
from torch import nn
from transformers import HubertModel, AutoModel, AutoConfig
from models.paths import resolve_model
from models.fusion import TriModalFusionEncoder
from models.video import VideoEncoder
from models.missing import encode_audio_with_pre_context_missing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _load_hubert(model_id_or_path: str) -> HubertModel:
    try:
        return HubertModel.from_pretrained(model_id_or_path)
    except (RuntimeError, ValueError):
        if not os.path.isdir(model_id_or_path):
            raise
        state_path = os.path.join(model_id_or_path, "pytorch_model.bin")
        if not os.path.exists(state_path):
            raise
        state = torch.load(state_path, map_location="cpu")
        g_key = "encoder.pos_conv_embed.conv.weight_g"
        v_key = "encoder.pos_conv_embed.conv.weight_v"
        if g_key in state and v_key in state:
            state["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = (
                state.pop(g_key)
            )
            state["encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = (
                state.pop(v_key)
            )
        config = AutoConfig.from_pretrained(model_id_or_path)
        model = HubertModel(config)
        model.load_state_dict(state, strict=False)
        return model


def _add_noise(x, sigma_max):
    if sigma_max <= 0:
        return x
    sigma = torch.rand(x.size(0), 1, device=x.device) * sigma_max
    return x + sigma * torch.randn_like(x)


def _extract_audio_features(
    backbone, audio_inputs, audio_mask, missing_owner=None, missing_key="audio"
):
    if missing_owner is not None and getattr(
        missing_owner, "_local_missing_spec", None
    ):
        hidden_states = encode_audio_with_pre_context_missing(
            missing_owner, backbone, audio_inputs, audio_mask, missing_key
        )
    else:
        audio_out = backbone(audio_inputs, audio_mask)
        hidden_states = audio_out.last_hidden_state
    with torch.no_grad():
        input_lengths = audio_mask.sum(-1)
        if hasattr(backbone, "_get_feat_extract_output_lengths"):
            feat_lengths = backbone._get_feat_extract_output_lengths(input_lengths)
        else:
            feat_lengths = input_lengths
        feat_lengths = feat_lengths.to(dtype=torch.long, device=hidden_states.device)
        feat_lengths = torch.clamp(feat_lengths, max=hidden_states.shape[1])
    seq_len = hidden_states.shape[1]
    arange = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
    audio_mask_new = (arange < feat_lengths.unsqueeze(1)).float()
    denom = feat_lengths.clamp(min=1).unsqueeze(1).to(hidden_states.dtype)
    features = (hidden_states * audio_mask_new.unsqueeze(-1)).sum(1) / denom
    return (hidden_states, features, audio_mask_new)


class ChineseModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.denoise = config.denoise
        self.denoise_sigma = config.denoise_sigma
        self.disable_unimodal_heads = bool(
            getattr(config, "disable_unimodal_heads", False)
        )
        self.roberta_model = AutoModel.from_pretrained(
            resolve_model("chinese-roberta-wwm-ext", "hfl/chinese-roberta-wwm-ext")
        )
        self.hubert_model = _load_hubert(
            resolve_model("chinese-hubert-base", "TencentGameMate/chinese-hubert-base")
        )
        self.video_encoder = VideoEncoder(config)
        self.trimodal_fusion = TriModalFusionEncoder(
            hidden_size=768,
            num_layers=config.num_hidden_layers,
            num_heads=12,
            bottleneck_tokens=getattr(config, "trimodal_bottleneck_tokens", 4),
            dropout=config.dropout,
            av_cross=getattr(config, "trimodal_av_cross", False),
            av_cross_weight=getattr(config, "trimodal_av_cross_weight", 0.2),
            tv_cross=getattr(config, "trimodal_tv_cross", False),
            tv_cross_weight=getattr(config, "trimodal_tv_cross_weight", 0.2),
            disable_bottleneck=getattr(config, "disable_bottleneck", False),
            disable_ta_cross=getattr(config, "disable_ta_cross", False),
            ta_to_a_only=getattr(config, "ta_to_a_only", False),
            bidirectional_ta_cross=getattr(config, "bidirectional_ta_cross", False),
            bidirectional_tv_cross=getattr(config, "bidirectional_tv_cross", False),
        )
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768, 1)
        )
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768, 1)
        )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.trimodal_fusion.out_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        if self.denoise:
            self.T_denoise = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
            )
            self.A_denoise = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
            )
            self.V_denoise = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
            )
            self.M_denoise = nn.Sequential(
                nn.Linear(self.trimodal_fusion.out_dim, self.trimodal_fusion.out_dim),
                nn.ReLU(),
                nn.Linear(self.trimodal_fusion.out_dim, self.trimodal_fusion.out_dim),
            )

    def forward(
        self, text_inputs, text_mask, audio_inputs, audio_mask, video_inputs, video_mask
    ):
        text_out = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = text_out.last_hidden_state
        T_features = text_out["pooler_output"]
        A_hidden_states, A_features, audio_mask_new = _extract_audio_features(
            self.hubert_model,
            audio_inputs,
            audio_mask,
            missing_owner=self,
            missing_key="audio",
        )
        V_hidden_states, V_features, video_mask_new = self.video_encoder(
            video_inputs, video_mask, missing_owner=self, missing_key="video"
        )
        fused_hidden_states = self.trimodal_fusion(
            T_hidden_states,
            text_mask,
            A_hidden_states,
            audio_mask_new,
            V_hidden_states,
            video_mask_new,
        )
        outputs = {"M": self.fused_output_layers(fused_hidden_states)}
        if not self.disable_unimodal_heads:
            outputs["T"] = self.T_output_layers(T_features)
            outputs["A"] = self.A_output_layers(A_features)
            outputs["V"] = self.V_output_layers(V_features)
        if self.denoise:
            outputs["M_clean"] = fused_hidden_states
            if not self.disable_unimodal_heads:
                outputs["T_clean"] = T_features
                outputs["A_clean"] = A_features
                outputs["V_clean"] = V_features
                outputs["T_denoised"] = self.T_denoise(
                    _add_noise(T_features, self.denoise_sigma)
                )
                outputs["A_denoised"] = self.A_denoise(
                    _add_noise(A_features, self.denoise_sigma)
                )
                outputs["V_denoised"] = self.V_denoise(
                    _add_noise(V_features, self.denoise_sigma)
                )
            outputs["M_denoised"] = self.M_denoise(
                _add_noise(fused_hidden_states, self.denoise_sigma)
            )
        return outputs
