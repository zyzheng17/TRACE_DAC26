# Contrastive Tasks

- `rtl/`: pairs original RTL graph cones with transformed positive RTL graph cones.
- `aig/`: learns aligned embeddings between paired AIG graphs.
- `pm/`: learns aligned embeddings between paired PM netlist graphs.

Evaluation entry points:

```bash
cd aig/src && python eval.py
cd ../../pm/src && python eval.py
cd ../../rtl/src && python eval.py --checkpoint ../../../../checkpoints/trace_contrastive_rtl_rec.ckpt
```
