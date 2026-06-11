# TRACE

TRACE organizes circuit learning code by data type and task type. It includes
contrastive and predictive learning workflows for RTL, AIG, and post-mapping
netlist graph datasets.

## Tutorial

For a small computational-graph example that introduces the core TRACE idea
without EDA dependencies, run:

```bash
cd tasks/tutorial/computational_graph
python demo.py
```

See [`computational_graph/README.md`](./computational_graph/README.md) for a
walkthrough of the tutorial.


## Code Layout

Data:
- `data/rtl`: RTL designs and graph datasets
- `data/aig`: AIG graph datasets
- `data/pm_netlist`: post-mapping netlist datasets

Tasks:
- `tasks/contrastive/rtl`: RTL pair contrastive learning
- `tasks/contrastive/aig`: AIG pair contrastive learning
- `tasks/contrastive/pm`: post-mapping pair contrastive learning
- `tasks/predictive/aig`: AIG probability/covariance prediction
- `tasks/predictive/pm`: post-mapping logic-1 probability prediction

Tutorial:
- `tasks/tutorial/computational_graph`: minimal computational-graph example

Large training artifacts should stay out of git. Keep downloaded or generated
datasets under `data/`; the largest contrastive `.pt` files can be symlinked to
external storage when local disk space is limited.

## Setup

We recommend using a conda environment, especially for GPU runs where PyTorch,
PyG, and DGL wheels must match the CUDA version.

```bash
conda create -n trace python=3.10
conda activate trace
pip install -r requirements.txt
```

`torch-geometric`, `torch-scatter`, and `dgl` are CUDA-version-specific. If you
plan to train on GPUs, install versions that match your local CUDA/PyTorch
setup. RTL preprocessing also requires external EDA tools such as Yosys, and
some AIG flows may require ABC/AIGER binaries.


## Dataset

The TRACE dataset is available on Hugging Face:
[`zyzheng23/TRACE_dataset`](https://huggingface.co/datasets/zyzheng23/TRACE_dataset).

Download the dataset and place it under `data/` with the following structure:

```text
data/
|-- aig/
|-- pm_netlist/
`-- rtl/
```


## Run Contrastive Tasks

The commands below use the default hyperparameters defined in each `train.py`.
Training uses GPU by default. Add `--use_cpu` only when running without a GPU.
Evaluation scripts expect the relevant checkpoints to be available under
`checkpoints/` or passed through each script's checkpoint argument.

### RTL

```bash
cd tasks/contrastive/rtl
python train.py
python eval.py
```

### AIG

```bash
cd tasks/contrastive/aig
python train.py
python eval.py
```

### Post-Mapping Netlist

```bash
cd tasks/contrastive/pm
python train.py
python eval.py
```

## Run Predictive Tasks

The predictive tasks train supervised models on labelled circuit graphs.

### AIG

```bash
cd tasks/predictive/aig
python train.py
python eval.py
```

Use `--learning_target covariance` to train the AIG predictive model on derived
AND-gate covariance labels.

### Post-Mapping Netlist

```bash
cd tasks/predictive/pm
python train.py
python eval.py
```
