import argparse
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader

from dataset.loader import parse_pm_graph
from models.trace_trainer import TRACETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TRACE on PM netlist predictive probability.')
    parser.add_argument('--data_path', type=Path, default=Path('../../../data/pm_netlist/iccad_dc_pm.npz'))
    parser.add_argument('--checkpoint', type=Path, default=Path('../../../checkpoints/trace_predictive_pm_probability.pth'))
    parser.add_argument('--trainval_split', default=0.9, type=float)
    parser.add_argument('--seed', default=208, type=int,
                        help='Validation split seed. Use the default to reproduce the reported result.')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--drop_last', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_rounds', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-10, type=float)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--max_batches', default=None, type=int,
                        help='Optional limit for quick smoke tests.')
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch. missing={missing}, unexpected={unexpected}')


def validation_graphs(data_path, trainval_split, seed):
    circuits = np.load(data_path, allow_pickle=True)['circuits'].item()
    names = list(circuits)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(names), generator=generator).tolist()
    cutoff = int(len(names) * trainval_split)
    valid_names = [names[index] for index in permutation[cutoff:]]
    return [parse_pm_graph(circuits[name]) for name in valid_names], len(names)


def main():
    args = parse_args()
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else f'cuda:{args.gpu_id}')
    model_args = Namespace(
        num_rounds=args.num_rounds,
        hidden=args.hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
    )
    model = TRACETrainer(model_args)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    graphs, total_count = validation_graphs(args.data_path, args.trainval_split, args.seed)
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False,
                        drop_last=args.drop_last, num_workers=args.num_workers)
    print(f'device {device}')
    print(f'total circuits {total_count} valid circuits {len(graphs)} batches {len(loader)}')

    maes, r2s = [], []
    all_target, all_pred = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            batch = batch.to(device)
            pred_cov = model(batch)
            pred_prob = model._cop_with_cov(batch, pred_cov)
            unique_targets = batch.forward_index[batch.forward_level > 0]
            target = batch.prob[unique_targets]
            pred = pred_prob[unique_targets]
            mae = torch.nn.functional.l1_loss(pred, target)
            r2 = r2_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())

            maes.append(float(mae.detach().cpu()))
            r2s.append(float(r2))
            all_target.append(target.detach().cpu())
            all_pred.append(pred.detach().cpu())
            print(f'batch {batch_idx + 1}/{len(loader)} mae {maes[-1]:.6f} r2 {r2s[-1]:.6f} '
                  f'running_mae {sum(maes) / len(maes):.6f} running_r2 {sum(r2s) / len(r2s):.6f}')

    all_target = torch.cat(all_target).numpy()
    all_pred = torch.cat(all_pred).numpy()
    global_mae = float(torch.nn.functional.l1_loss(torch.tensor(all_pred), torch.tensor(all_target)))
    global_r2 = float(r2_score(all_target, all_pred))
    print(f'RESULT batch_mae {sum(maes) / len(maes):.6f} batch_r2 {sum(r2s) / len(r2s):.6f} '
          f'global_mae {global_mae:.6f} global_r2 {global_r2:.6f}')


if __name__ == '__main__':
    main()
