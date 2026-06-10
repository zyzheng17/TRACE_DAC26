# Tasks

TRACE currently exposes two task families:

- `contrastive/`: learn aligned embeddings across equivalent or transformed circuit representations.
- `predictive/`: predict circuit-level or node-level labels such as signal probability and covariance.
- `tutorial/`: small self-contained experiments that explain the computational-graph motivation behind TRACE.

The contrastive and predictive task directories provide `train.py` and `eval.py` entry points. The `tutorial/` directory is a lightweight forward-pass demo for explaining graph construction and the TRACE encoder interface.
