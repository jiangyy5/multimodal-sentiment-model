import torch
from torch import nn
from transformers import RobertaModel, Data2VecAudioModel
from models.paths import resolve_model
from models.fusion import TriModalFusionEncoder
from models.video import VideoEncoder
from models.missing import encode_audio_with_pre_context_missing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class EnglishModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained(
            resolve_model("roberta-base", "roberta-base")
        )
        self.disable_unimodal_heads = bool(
            getattr(config, "disable_unimodal_heads", False)
        )
        self.video_encoder = VideoEncoder(config)
        self.data2vec_model = Data2VecAudioModel.from_pretrained(
            resolve_model("data2vec-audio-base", "facebook/data2vec-audio-base")
        )
        self.denoise = config.denoise
        self.denoise_sigma = config.denoise_sigma
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768 * 2, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768 * 2, 1)
        )
        self.v_cls_enable = bool(getattr(config, "v_cls_enable", True))
        self.v_cls_num_classes = int(getattr(config, "v_cls_num_classes", 7))
        self.V_output_layers = None
        self.V_cls_output_layers = None
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout), nn.Linear(768 * 2, 1)
        )
        if self.v_cls_enable:
            self.V_cls_output_layers = nn.Sequential(
                nn.Dropout(config.dropout), nn.Linear(768 * 2, self.v_cls_num_classes)
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
        fused_dim = self.trimodal_fusion.out_dim * 2
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(fused_dim, 768),
            nn.ReLU(),
            nn.Linear(768, 1),
        )
        if self.denoise:
            t_dim = self.T_output_layers[1].in_features
            a_dim = self.A_output_layers[1].in_features
            m_dim = fused_dim
            self.T_denoise = nn.Sequential(
                nn.Linear(t_dim, t_dim), nn.ReLU(), nn.Linear(t_dim, t_dim)
            )
            self.A_denoise = nn.Sequential(
                nn.Linear(a_dim, a_dim), nn.ReLU(), nn.Linear(a_dim, a_dim)
            )
            self.V_denoise = None
            self.V_denoise = nn.Sequential(
                nn.Linear(a_dim, a_dim), nn.ReLU(), nn.Linear(a_dim, a_dim)
            )
            self.M_denoise = nn.Sequential(
                nn.Linear(m_dim, m_dim), nn.ReLU(), nn.Linear(m_dim, m_dim)
            )

    def forward(
        self,
        text_inputs,
        text_mask,
        text_context_inputs,
        text_context_mask,
        audio_inputs,
        audio_mask,
        audio_context_inputs,
        audio_context_mask,
        video_inputs=None,
        video_mask=None,
        video_context_inputs=None,
        video_context_mask=None,
    ):
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)
        T_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]
        raw_output_context = self.roberta_model(
            text_context_inputs, text_context_mask, return_dict=True
        )
        T_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"]
        A_hidden_states, A_features, audio_mask_new = _extract_audio_features(
            self.data2vec_model,
            audio_inputs,
            audio_mask,
            missing_owner=self,
            missing_key="audio",
        )
        A_context_hidden_states, A_context_features, audio_context_mask_new = (
            _extract_audio_features(
                self.data2vec_model,
                audio_context_inputs,
                audio_context_mask,
                missing_owner=self,
                missing_key="audio_context",
            )
        )
        if (
            video_inputs is None
            or video_mask is None
            or video_context_inputs is None
            or (video_context_mask is None)
        ):
            raise ValueError(
                "trimodal=True requires video inputs for both current and context clips"
            )
        V_hidden_states, V_features, video_mask_new = self.video_encoder(
            video_inputs, video_mask, missing_owner=self, missing_key="video"
        )
        V_context_hidden_states, V_context_features, video_context_mask_new = (
            self.video_encoder(
                video_context_inputs,
                video_context_mask,
                missing_owner=self,
                missing_key="video_context",
            )
        )
        T_features = torch.cat((input_pooler, context_pooler), dim=1)
        A_features = torch.cat((A_features, A_context_features), dim=1)
        T_output = None
        A_output = None
        if not self.disable_unimodal_heads:
            T_output = self.T_output_layers(T_features)
            A_output = self.A_output_layers(A_features)
        V_output = None
        V_cls_logits = None
        V_features = torch.cat((V_features, V_context_features), dim=1)
        if not self.disable_unimodal_heads:
            V_output = self.V_output_layers(V_features)
        if not self.disable_unimodal_heads and self.V_cls_output_layers is not None:
            V_cls_logits = self.V_cls_output_layers(V_features)
        fused_current = self.trimodal_fusion(
            T_hidden_states,
            text_mask,
            A_hidden_states,
            audio_mask_new,
            V_hidden_states,
            video_mask_new,
        )
        fused_context = self.trimodal_fusion(
            T_context_hidden_states,
            text_context_mask,
            A_context_hidden_states,
            audio_context_mask_new,
            V_context_hidden_states,
            video_context_mask_new,
        )
        fused_hidden_states = torch.cat((fused_current, fused_context), dim=1)
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
