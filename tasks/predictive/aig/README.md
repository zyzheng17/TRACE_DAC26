# AIG Predictive Learning

```bash
python train.py --data_path ../../../data/aig/aig_predictive.pt --learning_target probability --use_cpu
```

Use `--learning_target covariance` to train against derived AND-gate covariance labels.
