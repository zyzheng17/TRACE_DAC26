import argparse
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from dataset.loader import OrderedData, load_train_valid_dataset  # noqa: F401 - required by torch.load pickles
from models.trace_trainer import TRACETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TRACE on PM netlist predictive labels.')
    parser.add_argument('--data_path', type=Path, default=Path('../../../data/pm_netlist/iccad_dc_pm.npz'),
                        help='Path to the PM netlist .npz file used by DeepCell-style preprocessing.')
    parser.add_argument('--work_dir', type=Path, default=Path('data/train'),
                        help='Directory for the processed PyG InMemoryDataset cache.')
    parser.add_argument('--log_dir', type=Path, default=Path('logs'))
    parser.add_argument('--trainval_split', default=0.9, type=float)
    parser.add_argument('--sample_ratio', default=1.0, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--max_epochs', default=40, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-10, type=float)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_rounds', default=1, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    train_dataset, valid_dataset = load_train_valid_dataset(
        work_dir=args.work_dir,
        npz_path=args.data_path,
        trainval_split=args.trainval_split,
        seed=args.seed,
        sample_ratio=args.sample_ratio,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              drop_last=True, num_workers=args.num_workers)
    model = TRACETrainer(args)
    accelerator = 'cpu' if args.use_cpu else 'gpu'
    devices = 1 if args.use_cpu else [args.gpu_id]
    trainer = pl.Trainer(default_root_dir=str(args.log_dir), max_epochs=args.max_epochs,
                         accelerator=accelerator, devices=devices, log_every_n_steps=10)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()
