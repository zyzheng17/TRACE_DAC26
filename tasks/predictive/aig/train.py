import argparse
from pathlib import Path

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from dataset.loader import GraphDataset, OrderedData  # noqa: F401 - required by torch.load pickles
from models.trace_trainer import TRACETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TRACE on AIG predictive labels.')
    parser.add_argument('--data_path', type=Path, default=Path('../../../data/aig/aig_predictive.pt'),
                        help='Path to a torch .pt file containing AIG graph objects.')
    parser.add_argument('--val_names', nargs='*', default=['b12_opt_C', 'b14_opt_C'],
                        help='Circuit names reserved for validation.')
    parser.add_argument('--log_dir', type=Path, default=Path('logs'))
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--learning_target', default='probability', choices=['probability', 'covariance'])
    parser.add_argument('--pool', default='mean')
    parser.add_argument('--in_channels', default=4, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train_dataset = GraphDataset(args.data_path, is_train=True, val_list=args.val_names)
    val_dataset = GraphDataset(args.data_path, is_train=False, val_list=args.val_names)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)

    model = TRACETrainer(args)
    accelerator = 'cpu' if args.use_cpu else 'gpu'
    devices = 1 if args.use_cpu else [args.gpu_id]
    trainer = pl.Trainer(default_root_dir=str(args.log_dir), max_epochs=args.max_epochs,
                         accelerator=accelerator, devices=devices, log_every_n_steps=100)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
