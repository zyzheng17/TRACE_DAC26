# RTL Contrastive Learning

Run from `src/` so local imports resolve:

```bash
cd src
python train.py --data_root ../../../../data/rtl/dataset_graph/data_bench --use_cpu
```

Data preprocessing lives in `data_prep/prepare_rtl_dataset.py` and expects raw RTL designs plus Yosys for transformation.
