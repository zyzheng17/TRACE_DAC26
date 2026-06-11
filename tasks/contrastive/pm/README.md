# PM Netlist Contrastive Learning

This task trains TRACE on paired PM netlist graphs.

```bash
python train.py \
  --data_path ../../../data/pm_netlist/<prefix> \
  --use_cpu
```

The loader expects `<prefix>_train.pt` and `<prefix>_test.pt` unless `--data_path` points directly to a `.pt` file.
