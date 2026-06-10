# TRACE

TRACE organizes circuit-learning code by data type and task type.

## Code Layout

- `data/rtl`: raw RTL designs and generated RTL graph datasets.
- `data/aig`: AIG graph datasets for contrastive and predictive learning.
- `data/pm_netlist`: PM netlist graph datasets.
- `tasks/contrastive/rtl`: RTL contrastive learning from original and transformed RTL graph pairs.
- `tasks/contrastive/aig`: AIG contrastive learning from paired graph datasets.
- `tasks/contrastive/pm`: PM netlist contrastive learning from paired graph datasets.
- `tasks/predictive/aig`: AIG predictive learning for signal probability or covariance targets.
- `tasks/predictive/pm`: PM netlist predictive learning for logic-1 probability targets.
- `tasks/tutorial/computational_graph`: a minimal tutorial setting for computational graph reasoning with add/sub/mul modular expressions.

Large training artifacts should stay out of git. This working tree keeps the recovered datasets under `data/`; the largest contrastive `.pt` files may be symlinks to the original storage location when local disk space is limited.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`torch-geometric`, `torch-scatter`, and `dgl` are CUDA-version-specific; install wheels matching your CUDA/PyTorch version when using GPUs. RTL preprocessing also needs external EDA tools such as Yosys, and some AIG flows may need ABC/AIGER binaries.

## Run Contrastive Tasks

RTL contrastive training:

```bash
cd tasks/contrastive/rtl/src
python train.py \
  --data_root ../../../../data/rtl/dataset_graph/data_bench \
  --log_dir logs \
  --batch_size 2 \
  --max_epochs 1 \
  --use_cpu
```

AIG contrastive training:

```bash
cd tasks/contrastive/aig/src
python train.py \
  --data_path ../../../../data/aig/forgeeda_pm_aig \
  --batch_size 1024 \
  --max_epochs 1 \
  --use_cpu
```

PM netlist contrastive training:

```bash
cd tasks/contrastive/pm/src
python train.py \
  --data_path ../../../../data/pm_netlist/forgeeda_pm_pair_pkl \
  --batch_size 1024 \
  --max_epochs 1 \
  --use_cpu
```

## Run Predictive Tasks

AIG predictive training:

```bash
cd tasks/predictive/aig
python train.py \
  --data_path ../../../data/aig/aig_predictive.pt \
  --learning_target probability \
  --batch_size 1 \
  --max_epochs 1 \
  --use_cpu
```

PM netlist predictive training:

```bash
cd tasks/predictive/pm
python train.py \
  --data_path ../../../data/pm_netlist/iccad_dc_pm.npz \
  --work_dir data/train \
  --batch_size 512 \
  --max_epochs 1 \
  --use_cpu
```

## Run Eval

AIG contrastive retrieval:

```bash
cd tasks/contrastive/aig/src
python eval.py
```

PM netlist contrastive retrieval:

```bash
cd tasks/contrastive/pm/src
python eval.py
```

PM netlist predictive probability:

```bash
cd tasks/predictive/pm
python eval.py
```

The same eval entry points also exist for `tasks/contrastive/rtl/src` and `tasks/predictive/aig`; they are ready to use once the corresponding checkpoints are added under `checkpoints/`.

## Rebuild RTL Data

```bash
cd tasks/contrastive/rtl
python data_prep/prepare_rtl_dataset.py \
  --designs ../../../../data/rtl/designs.json \
  --ori-dir ../../../../data/rtl/raw \
  --work-dir ../../../../data/rtl/work \
  --dataset-out ../../../../data/rtl/dataset_graph/data_bench \
  --split-dir ../../../../data/rtl/splits \
  --yosys-bin "$YOSYS_BIN"
```

Use `--stages` to run part of the RTL pipeline only: `registers`, `transform`, `whole_graph`, `cones`, `cone_graph`, or `dataset`.

## Tutorial

For a small computational-graph example that explains the core TRACE idea without EDA dependencies:

```bash
cd tasks/tutorial/computational_graph
python demo.py
```

See `tasks/tutorial/computational_graph/README.md` for the tutorial narrative and paper figures.

## Paper Settings

The public entry points expose only the TRACE encoder. Task-level Lightning modules are named `TRACETrainer` and live in `trace_trainer.py`, while the encoder is exposed as `TRACE` through `trace_encoder.py`. The paper reports contrastive retrieval with Rec@1/5/10 on RTL, AIG, and PM netlists, and predictive MAE/R2 on AIG and PM netlists. See `checkpoints/README.md` for the checkpoint naming convention used for each setting.

See `REPRODUCIBILITY.md` for the paper target metrics and the current reproduction status.
