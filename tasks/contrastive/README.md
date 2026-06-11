# Contrastive Tasks

Contrastive tasks train TRACE encoders to align original circuits with positive transformed variants.

- `aig/`: paired AIG graph retrieval.
- `pm/`: paired PM netlist graph retrieval.
- `rtl/`: paired RTL graph-cone retrieval.

Each task keeps only the files needed to run that variant:

```text
<task>/
  train.py
  eval.py
  dataset/
  models/
```

Evaluation entry points:

```bash
cd aig && python eval.py
cd ../pm && python eval.py
cd ../rtl && python eval.py --checkpoint ../../../checkpoints/trace_contrastive_rtl_rec.ckpt
```
