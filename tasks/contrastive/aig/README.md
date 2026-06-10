# AIG Contrastive Learning

This task trains TRACE on paired AIG graphs, such as original and synthesized AIG variants.

```bash
cd src
python train.py \
  --data_path ../../../../data/aig/<prefix> \
  --use_cpu
```

The loader expects `<prefix>_train.pt` and `<prefix>_test.pt` unless `--data_path` points directly to a `.pt` file.
