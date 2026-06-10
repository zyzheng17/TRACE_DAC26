# Reproducibility Notes

This repository is organized around the settings in the TRACE paper:

- Contrastive retrieval on RTL, AIG, and PM netlist graphs, evaluated by Rec@1/5/10.
- Predictive learning on AIG and PM netlist graphs, evaluated by MAE and R2.

## Paper Targets

Contrastive TRACE results reported in the paper:

- RTL: Rec@1 94.45, Rec@5 98.74, Rec@10 99.89.
- AIG: Rec@1 92.68, Rec@5 98.65, Rec@10 99.51.
- PM netlist: Rec@1 90.81, Rec@5 98.48, Rec@10 99.44.

Predictive TRACE results reported in the paper:

- Combinational AIG probability: R2 0.989, MAE 0.015.
- Combinational AIG similarity: R2 0.633, MAE 0.055.
- Sequential AIG probability: R2 0.997, MAE 0.009.
- Sequential AIG transition probability: R2 0.976, MAE 0.005.
- PM netlist probability: R2 0.994, MAE 0.013.

## Current Repository Status

The source tree currently exposes only the TRACE encoder in each task entry point. Task-level Lightning modules are named `TRACETrainer` and live in `trace_trainer.py`:

- `tasks/contrastive/rtl/src/train.py`
- `tasks/contrastive/aig/src/train.py`
- `tasks/contrastive/pm/src/train.py`
- `tasks/predictive/aig/train.py`
- `tasks/predictive/pm/train.py`

Available trained checkpoints:

- `checkpoints/trace_contrastive_aig_rec.ckpt`: AIG contrastive retrieval checkpoint from `Function-Contrastive-Learning/AIG_logs/lightning_logs/version_3`. Its validation Rec@1/5/10 are 92.68/98.65/99.51, matching the paper's AIG contrastive target.
- `checkpoints/trace_contrastive_pm_rec.ckpt`: PM netlist contrastive retrieval checkpoint from `Function-Contrastive-Learning/lightning_logs/lightning_logs/version_9`. Its validation Rec@1/5/10 are 90.81/98.48/99.44, matching the paper's PM contrastive target.
- `checkpoints/trace_predictive_pm_probability.pth`: PM netlist predictive checkpoint for the DeepCell_DAC covariance-residual path. It loads into the cleaned PM TRACE encoder without missing or unexpected parameters.

RTL contrastive and predictive AIG paper settings still need checkpoints. See `checkpoints/README.md` for the naming scheme.

Recovered local datasets are listed in `data/README.md`. The large AIG and PM contrastive `.pt` files are symlinked into `data/` because the local disk does not have enough free space for duplicate copies.

## Validation Attempt

The source tree passes Python syntax compilation with `python -m compileall -q TRACE`.

Running the task entry points with `--help` is currently blocked by the local Python environment before reaching TRACE code: `pytorch_lightning` fails because the installed `torch` package is incomplete and does not expose `torch.Tensor`. Reinstall PyTorch for the target CUDA/CPU environment before running training or evaluation.

Inference was re-run locally with the `fgnn` and `graphrag` conda environments because the base environment's PyTorch install is incomplete:

- AIG contrastive, `trace_contrastive_aig_rec.ckpt` on `data/aig/forgeeda_pm_aig_test.pt`: Rec@1 93.24, Rec@5 99.06, Rec@10 99.62.
- PM netlist contrastive, `trace_contrastive_pm_rec.ckpt` on `data/pm_netlist/forgeeda_pm_pair_pkl_test.pt`: Rec@1 91.29, Rec@5 98.48, Rec@10 99.49.
- PM netlist predictive now follows `DeepCell_DAC`'s covariance-residual evaluation: TRACE predicts covariance residuals during training, and validation reconstructs node probabilities with cumulative COP. `trace_predictive_pm_probability.pth` loads with 0 missing and 0 unexpected parameters. Full validation eval with seed 208, batch size 128, and `drop_last=True` gives global MAE 0.01255 and global R2 0.99388, matching the paper target.

## Reproduction Gaps

The current code can train the available TRACE task entry points, but it cannot fully reproduce the paper results without additional assets:

- Required processed RTL contrastive pickle datasets are not present.
- Trained checkpoints for RTL contrastive and predictive AIG are not present.
- Evaluation-only scripts that load a checkpoint and compute the exact paper Rec@k, MAE, and R2 tables still need to be added.
- Sequential AIG predictive entry points are not yet separated in this cleaned source tree.
- The current Python environment must be repaired because PyTorch import is incomplete.

After datasets and checkpoints are added, reproduction should be verified by running each task with the corresponding checkpoint and comparing against the targets above.
