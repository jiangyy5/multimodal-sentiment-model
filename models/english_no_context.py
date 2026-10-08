import torch
from torch import nn
from transformers import RobertaModel, Data2VecAudioModel
from models.paths import resolve_model
from models.fusion import TriModalFusionEncoder
from models.video import VideoEncoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _add_noise(x, sigma_max):
    if sigma_max <= 0:
        return x
    sigma = torch.rand(x.size(0), 1, device=x.device) * sigma_max
    return x + sigma * torch.randn_like(x)


def _extract_audio_features(backbone, audio_inputs, audio_mask):
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


class EnglishModelWithoutContext(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.denoise = config.denoise
        self.denoise_sigma = config.denoise_sigma
        self.disable_unimodal_heads = bool(
            getattr(config, "disable_unimodal_heads", False)
        )
        self.roberta_model = RobertaModel.from_pretrained(
            resolve_model("roberta-base", "roberta-base")
        )
        self.video_encoder = VideoEncoder(config)
        self.data2vec_model = Data2VecAudioModel.from_pretrained(
            resolve_model("data2vec-audio-base", "facebook/data2vec-audio-base")
        )
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768, 1)
        )
        self.a_cls_enable = getattr(config, "a_cls_enable", False)
        self.a_cls_num_classes = int(getattr(config, "a_cls_num_classes", 7))
        if self.a_cls_enable:
            self.A_cls_output_layers = nn.Sequential(
                nn.Dropout(config.dropout), nn.Linear(768, self.a_cls_num_classes)
            )
        self.v_cls_enable = bool(getattr(config, "v_cls_enable", True))
        self.v_cls_num_classes = int(getattr(config, "v_cls_num_classes", 7))
        self.V_output_layers = None
        self.V_cls_output_layers = None
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768, 1)
        )
        if self.v_cls_enable:
            self.V_cls_output_layers = nn.Sequential(
                nn.Dropout(config.dropout), nn.Linear(768, self.v_cls_num_classes)
            )
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
        fused_dim = self.trimodal_fusion.out_dim
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(fused_dim, 768),
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
            self.V_denoise = None
            self.V_denoise = nn.Sequential(
                nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768)
            )
            self.M_denoise = nn.Sequential(
                nn.Linear(fused_dim, fused_dim),
                nn.ReLU(),
                nn.Linear(fused_dim, fused_dim),
            )

    def forward(
        self,
        text_inputs,
        text_mask,
        audio_inputs,
        audio_mask,
        video_inputs=None,
        video_mask=None,
    ):
        A_hidden_states, A_features, audio_mask_new = _extract_audio_features(
            self.data2vec_model, audio_inputs, audio_mask
        )
        if video_inputs is None or video_mask is None:
            raise ValueError("trimodal=True requires video_inputs and video_mask")
        V_hidden_states, V_features, video_mask_new = self.video_encoder(
            video_inputs, video_mask
        )
        A_output = None
        A_cls_logits = None
        V_output = None
        V_cls_logits = None
        if not self.disable_unimodal_heads:
            A_output = self.A_output_layers(A_features)
            A_cls_logits = (
                self.A_cls_output_layers(A_features) if self.a_cls_enable else None
            )
            V_output = self.V_output_layers(V_features)
            V_cls_logits = (
                self.V_cls_output_layers(V_features)
                if self.V_cls_output_layers is not None
                else None
            )
        raw_output = self.roberta_model(text_inputs, text_mask)
        T_hidden_states = raw_output.last_hidden_state
        T_features = raw_output["pooler_output"]
        T_output = None
        if not self.disable_unimodal_heads:
            T_output = self.T_output_layers(T_features)
        fused_hidden_states = self.trimodal_fusion(
            T_hidden_states,
            text_mask,
            A_hidden_states,
            audio_mask_new,
            V_hidden_states,
            video_mask_new,
        )
        fused_output = self.fused_output_layers(fused_hidden_states)
        outputs = {"M": fused_output}
        if T_output is not None:
            outputs["T"] = T_output
        if A_output is not None:
            outputs["A"] = A_output
        if V_output is not None:
            outputs["V"] = V_output
        if V_cls_logits is not None:
            outputs["V_cls_logits"] = V_cls_logits
        if A_cls_logits is not None:
            outputs["A_cls_logits"] = A_cls_logits
        if self.denoise:
            outputs["M_clean"] = fused_hidden_states
            if not self.disable_unimodal_heads:
                outputs["T_clean"] = T_features
                outputs["A_clean"] = A_features
                outputs["T_denoised"] = self.T_denoise(
                    _add_noise(T_features, self.denoise_sigma)
                )
                outputs["A_denoised"] = self.A_denoise(
                    _add_noise(A_features, self.denoise_sigma)
                )
                outputs["V_clean"] = V_features
                outputs["V_denoised"] = self.V_denoise(
                    _add_noise(V_features, self.denoise_sigma)
                )
            outputs["M_denoised"] = self.M_denoise(
                _add_noise(fused_hidden_states, self.denoise_sigma)
            )
        return outputs
