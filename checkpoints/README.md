# TRACE Checkpoints

## Available Checkpoints

- `trace_contrastive_aig_rec.ckpt`
  - Source: `Function-Contrastive-Learning/AIG_logs/lightning_logs/version_3/checkpoints/epoch=99-step=3900.ckpt`
  - Setting: AIG contrastive retrieval.
  - Hparams: `encoder_type=TRACE` in the cleaned code (`CustomGNN` in the original log), `data_path=forgeeda_pm_aig`, `batch_size=1024`, `hidden=128`, `num_layers=9`.
  - Final validation metrics: Rec@1 92.68, Rec@5 98.65, Rec@10 99.51.
- `trace_contrastive_pm_rec.ckpt`
  - Source: `Function-Contrastive-Learning/lightning_logs/lightning_logs/version_9/checkpoints/epoch=99-step=4400.ckpt`
  - Setting: PM netlist contrastive retrieval.
  - Hparams: `encoder_type=TRACE` in the cleaned code (`CustomGNN` in the original log), `data_path=forgeeda_pm_pair_pkl`, `batch_size=1024`, `hidden=128`, `num_layers=9`.
  - Final validation metrics: Rec@1 90.81, Rec@5 98.48, Rec@10 99.44.
- `trace_predictive_pm_probability.pth`
  - Source: `TRACE/model_last_pm.pth` during local recovery; organized here as the PM predictive checkpoint.
  - Setting: PM netlist predictive probability with the `DeepCell_DAC` covariance-residual objective.
  - Current inference status: loads into the cleaned PM TRACE encoder with 0 missing and 0 unexpected parameters. Full validation eval with seed 208, batch size 128, and `drop_last=True` gives global MAE 0.01255 and global R2 0.99388.

## Naming Convention

Use names that encode the paper setting:

- `trace_contrastive_rtl_rec.ckpt`
- `trace_contrastive_aig_rec.ckpt`
- `trace_contrastive_pm_rec.ckpt`
- `trace_predictive_aig_probability.ckpt`
- `trace_predictive_aig_similarity.ckpt`
- `trace_predictive_aig_transition.ckpt`
- `trace_predictive_pm_probability.pth`

Keep large checkpoint files in release assets or Git LFS for open-source distribution.
