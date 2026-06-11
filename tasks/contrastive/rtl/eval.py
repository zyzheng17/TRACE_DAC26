import argparse
from pathlib import Path

import torch

from dataset.paired_dataloader import PairedDataLoader
from dataset.loader import OrderedData, load_train_valid_dataset_stage_align  # noqa: F401 - required by torch.load pickles
from models.trace_trainer import TRACETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TRACE on RTL contrastive retrieval.')
    parser.add_argument('--data_root', default='../../../data/rtl/dataset_graph/data_bench', type=str,
                        help='Directory containing dataset_{train,valid}_{ori,pos}.pkl')
    parser.add_argument('--checkpoint', type=Path, default=Path('../../../checkpoints/trace_contrastive_rtl_rec.ckpt'))
    parser.add_argument('--split', default='valid', choices=['train', 'valid'])
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--pool', default='PO', type=str)
    parser.add_argument('--in_channels', default=16, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--num_layers', default=9, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--max_batches', default=None, type=int,
                        help='Optional limit for quick smoke tests.')
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    mapped = {}
    for key, value in state.items():
        if key.startswith('encoder.'):
            mapped[key] = value
        elif key.startswith('rtl_encoder.'):
            mapped[key.replace('rtl_encoder.', 'encoder.', 1)] = value
    if not mapped:
        mapped = state
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        raise RuntimeError(f'Missing checkpoint parameters: {missing}')
    if unexpected:
        print(f'Ignored unexpected checkpoint parameters: {unexpected}')


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')

    ori_loader, pos_loader = load_train_valid_dataset_stage_align(
        batch_size=args.batch_size,
        train_valid=args.split,
        data_root=args.data_root,
    )
    loader = PairedDataLoader(ori_loader, pos_loader, encoder_type='TRACE')
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else f'cuda:{args.gpu_id}')

    model = TRACETrainer(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    losses, r1s, r5s, r10s = [], [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            loss, metrics = model.forward_cls(batch)
            losses.append(float(loss.detach().cpu()))
            r1s.append(metrics['R@1'])
            r5s.append(metrics['R@5'])
            r10s.append(metrics['R@10'])
            print(f'batch {batch_idx + 1} loss {losses[-1]:.4f} '
                  f'R@1 {r1s[-1]:.4f} R@5 {r5s[-1]:.4f} R@10 {r10s[-1]:.4f}')

    print(f'RESULT loss {sum(losses) / len(losses):.6f} '
          f'R@1 {sum(r1s) / len(r1s):.6f} '
          f'R@5 {sum(r5s) / len(r5s):.6f} '
          f'R@10 {sum(r10s) / len(r10s):.6f}')


if __name__ == '__main__':
    main()
