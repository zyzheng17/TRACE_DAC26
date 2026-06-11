import argparse

import pytorch_lightning as pl

from dataset.paired_dataloader import PairedDataLoader
from dataset.loader import OrderedData, load_train_valid_dataset_stage_align  # noqa: F401 - required by torch.load pickles
from models.trace_trainer import TRACETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train TRACE on RTL contrastive pairs.')
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--data_root', default='../../../data/rtl/dataset_graph/data_bench', type=str,
                        help='Directory containing dataset_{train,valid}_{ori,pos}.pkl')
    parser.add_argument('--log_dir', default='logs', type=str,
                        help='Directory for PyTorch Lightning logs and checkpoints')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_epochs', default=500, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--pool', default='PO', type=str)
    parser.add_argument('--in_channels', default=16, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--num_layers', default=9, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train_ori_loader, train_pos_loader = load_train_valid_dataset_stage_align(
        batch_size=args.batch_size,
        train_valid='train',
        data_root=args.data_root,
    )
    valid_ori_loader, valid_pos_loader = load_train_valid_dataset_stage_align(
        batch_size=args.batch_size,
        train_valid='valid',
        data_root=args.data_root,
    )

    train_loader = PairedDataLoader(train_ori_loader, train_pos_loader, encoder_type='TRACE')
    valid_loader = PairedDataLoader(valid_ori_loader, valid_pos_loader, encoder_type='TRACE')
    model = TRACETrainer(args)

    accelerator = 'cpu' if args.use_cpu else 'gpu'
    devices = 1 if args.use_cpu else [args.gpu_id]
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        max_epochs=args.max_epochs,
        devices=devices,
        accelerator=accelerator,
        log_every_n_steps=10,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()
