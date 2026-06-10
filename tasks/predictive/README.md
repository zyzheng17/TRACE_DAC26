# Predictive Tasks

Predictive tasks train supervised TRACE models on labelled circuit graphs.

- `aig/`: AIG probability/covariance prediction.
- `pm/`: PM netlist logic-1 probability prediction with function shift learning.

Evaluation entry points:

```bash
cd pm && python eval.py
cd ../aig && python eval.py --checkpoint ../../../checkpoints/trace_predictive_aig_probability.ckpt
```
