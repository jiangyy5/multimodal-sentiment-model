# Training options

Run one seed at a time with `python train.py --config configs/mosi.json --seed 1`.
Common defaults are in `configs/defaults.json`; dataset files contain only overrides.
An unknown parameter name is rejected. Use `--dry-run` to inspect the effective parameters.

The implementation uses text, audio and video throughout. It includes English models with and
without context, and a Chinese model without context.

Useful configuration keys:

| Key | Meaning |
|---|---|
| `batch_size`, `lr`, `grad_accum_steps` | Microbatch size, learning rate, gradient accumulation |
| `context`, `text_context_len`, `audio_context_len` | English context; audio/video share the latter length |
| `video_frames`, `video_chunk_size` | Frames per clip and frame-encoding chunk size |
| `video_stage1_epochs`, `video_stage2_unfreeze_last_n` | Visual backbone freeze/unfreeze schedule |
| `disable_bottleneck`, `disable_ta_cross`, `trimodal_tv_cross` | Fusion ablation controls |
| `bidirectional_ta_cross`, `bidirectional_tv_cross` | Optional reverse direct-attention paths |
| `m_only_supervision` | Supervise M only while retaining the forward structure |
| `denoise`, `denoise_weight`, `denoise_sigma` | Feature-denoising auxiliary objective |

SIMS does not use English context or English classification/correlation auxiliary settings.
Those shared defaults are not passed to the Chinese trainer. English branches share the multimodal
label; SIMS uses its provided M/T/A/V labels.

The complete-modality trainer saves separate checkpoints selected by validation accuracy and validation
total loss, and tests both at the end of training. Each new run receives its own directory under `runs/`.
No cross-run statistics are computed by the launcher.

Implementation behavior to keep in mind:

- AdamW is created per epoch in complete-modality training.
- Validation total loss includes the stochastic denoising objective when it is enabled.
- The English loop stops by `early_stop`; its `epochs` parameter is not a hard cap. SIMS applies both.
- Pearson loss is zero for microbatch size one; gradient accumulation does not form a larger Pearson batch.
- Video normalization depends on the downloaded processor configuration. Supply the complete model asset folder.

## Random-local-missing training

The existing missing-input training routine is available separately:

```bash
python train_missing.py --dataset mosi --train --model_seed 1 --selection_split valid --save_dir runs/missing-mosi-seed1
```

It reads the same dataset and pretrained-model directories. Use a new `--save_dir` per run.
Inspect available training arguments with `python train_missing.py --help`.
Audio/video features are erased before temporal context modeling. SIMS training removes its known
cross-split duplicate by default in this routine; the complete-modality trainer retains the supplied split.
This difference is controlled by `--keep_sims_train_test_duplicate` in the missing-input routine.

Post-training missing-input evaluation saves per-checkpoint, per-missing-rate records as
`missing_eval_seed<seed>.csv` and `.json` inside the run directory. It does not generate
paper tables or combine seeds. Use `--skip_post_eval` to skip that evaluation.
Generated histories, results and checkpoints are local outputs excluded from Git.
