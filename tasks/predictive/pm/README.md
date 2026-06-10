# PM Netlist Predictive Learning

This task trains TRACE on PM netlist probability prediction using the DeepCell-style `.npz` dataset format.

```bash
python train.py \
  --data_path ../../../data/pm_netlist/iccad_dc_pm.npz \
  --work_dir data/train \
  --batch_size 512 \
  --max_epochs 200 \
  --use_cpu
```

The model learns the function shift/covariance target during training and reconstructs logic-1 probability during validation and test.
