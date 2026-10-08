"""Transient feature-level missingness used by robustness experiments.

This module adds no parameters and therefore does not change checkpoint keys.
The evaluation/training driver attaches ``_local_missing_spec`` to a model for
one forward pass. Audio/video local sequence features are zeroed before their
contextual temporal Transformers, preventing retained tokens from carrying
information from erased time steps.
"""

import torch

KEY_OFFSETS = {"audio": 17, "audio_context": 19, "video": 23, "video_context": 29}


def _stable_seed(base_seed, sample_index, key):
    return (int(base_seed) * 1000003 + int(sample_index) * 9176 + KEY_OFFSETS[key]) % (
        2**63 - 1
    )


def select_bernoulli_subset(positions, rate, seed):
    """Select positions independently with probability ``rate``.

    Reusing the same seed across rates makes the masks nested while retaining
    LNLN's per-position Bernoulli missing-data mechanism.
    """
    count = int(positions.numel())
    if count == 0 or float(rate) <= 0:
        return positions[:0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected_offsets = (
        (torch.rand(count, generator=generator) < float(rate))
        .nonzero(as_tuple=False)
        .flatten()
    )
    return positions.detach().cpu()[selected_offsets]


def apply_feature_missing(owner, sequence, valid_mask, modality, key):
    spec = getattr(owner, "_local_missing_spec", None)
    if not spec:
        return sequence
    rates = spec.get(f"{modality}_rates")
    sample_indices = spec.get("sample_indices")
    if rates is None or sample_indices is None:
        return sequence
    output = sequence.clone()
    rates = rates.view(-1).detach().cpu()
    sample_indices = sample_indices.view(-1).detach().cpu()
    mask = (
        valid_mask.bool()
        if valid_mask is not None
        else torch.ones(sequence.shape[:2], dtype=torch.bool, device=sequence.device)
    )
    for row in range(sequence.size(0)):
        positions = mask[row].nonzero(as_tuple=False).flatten()
        selected = select_bernoulli_subset(
            positions,
            float(rates[row]),
            _stable_seed(spec["mask_seed"], int(sample_indices[row]), key),
        )
        if selected.numel() > 0:
            output[row, selected.to(sequence.device)] = 0
    return output


def masked_mean(sequence, valid_mask):
    if valid_mask is None:
        return sequence.mean(dim=1)
    weights = valid_mask.to(device=sequence.device, dtype=sequence.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp(min=1.0)
    return (sequence * weights).sum(dim=1) / denominator


def encode_audio_with_pre_context_missing(
    owner, backbone, audio_inputs, audio_mask, key
):
    """Run HuBERT/Data2Vec with erasure before the contextual encoder.

    Hugging Face's audio models first produce local convolutional features and
    then contextualize them with a Transformer. Erasing the final hidden states
    would allow every retained state to contain information from erased time
    steps. This helper mirrors the upstream forward pass and performs erasure
    immediately after feature projection and before ``backbone.encoder``.
    """
    extract_features = backbone.feature_extractor(audio_inputs).transpose(1, 2)
    feature_mask = None
    if audio_mask is not None:
        try:
            feature_mask = backbone._get_feature_vector_attention_mask(
                extract_features.shape[1], audio_mask, add_adapter=False
            )
        except TypeError:
            feature_mask = backbone._get_feature_vector_attention_mask(
                extract_features.shape[1], audio_mask
            )
    projected = backbone.feature_projection(extract_features)
    hidden_states = projected[0] if isinstance(projected, tuple) else projected
    hidden_states = apply_feature_missing(
        owner, hidden_states, feature_mask, "audio", key
    )
    try:
        hidden_states = backbone._mask_hidden_states(
            hidden_states, mask_time_indices=None, attention_mask=feature_mask
        )
    except TypeError:
        hidden_states = backbone._mask_hidden_states(
            hidden_states, mask_time_indices=None
        )
    encoder_outputs = backbone.encoder(
        hidden_states,
        attention_mask=feature_mask,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    hidden_states = encoder_outputs[0]
    adapter = getattr(backbone, "adapter", None)
    if adapter is not None:
        hidden_states = adapter(hidden_states)
    return hidden_states
