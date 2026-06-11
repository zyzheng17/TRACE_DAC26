# RTL Contrastive Learning

This task consumes preprocessed RTL graph-pair pickles named `dataset_{train,valid}_{ori,pos}.pkl`.

```bash
python train.py --data_root ../../../data/rtl/dataset_graph/data_bench --use_cpu
```

Pass `--data_root` either to that pair directory directly or to a parent containing `dataset_graph/data_bench`.
