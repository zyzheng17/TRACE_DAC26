import argparse
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from dataset.loader import GraphDataset, OrderedData  # noqa: F401 - required by torch.load pickles
from models.trace_trainer import TRACETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TRACE on AIG contrastive pairs.')
    parser.add_argument('--data_path', type=Path, default=Path('../../../data/aig/forgeeda_pm_aig'),
                        help='Prefix of *_train.pt and *_test.pt, or full path without the split suffix.')
    parser.add_argument('--log_dir', type=Path, default=Path('logs'))
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--pool', default='PO')
    parser.add_argument('--in_channels', default=3, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--num_layers', default=9, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    return parser.parse_args()


def split_path(prefix: Path, split: str) -> Path:
    prefix = prefix.expanduser()
    if prefix.suffix == '.pt':
        return prefix
    return prefix.with_name(prefix.name + f'_{split}.pt')


def main():
    args = parse_args()
    train_dataset = GraphDataset(split_path(args.data_path, 'train'))
    val_dataset = GraphDataset(split_path(args.data_path, 'test'))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                              drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=False, num_workers=args.num_workers)

    model = TRACETrainer(args)
    accelerator = 'cpu' if args.use_cpu else 'gpu'
    devices = 1 if args.use_cpu else [args.gpu_id]
    trainer = pl.Trainer(default_root_dir=str(args.log_dir), max_epochs=args.max_epochs,
                         accelerator=accelerator, devices=devices, log_every_n_steps=10)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
