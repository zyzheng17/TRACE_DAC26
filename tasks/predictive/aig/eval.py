import argparse
from pathlib import Path

import torch
from sklearn.metrics import r2_score
from torch_geometric.loader import DataLoader

from dataset.loader import GraphDataset, OrderedData  # noqa: F401 - required by torch.load pickles
from models.cov2prob import cov2prob
from models.trace_trainer import TRACETrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate TRACE on AIG predictive labels.')
    parser.add_argument('--data_path', type=Path, default=Path('../../../data/aig/aig_predictive.pt'))
    parser.add_argument('--checkpoint', type=Path, default=Path('../../../checkpoints/trace_predictive_aig_probability.ckpt'))
    parser.add_argument('--val_names', nargs='*', default=['b12_opt_C', 'b14_opt_C'])
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--learning_target', default='probability', choices=['probability', 'covariance'])
    parser.add_argument('--pool', default='mean')
    parser.add_argument('--in_channels', default=4, type=int)
    parser.add_argument('--hidden', default=128, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint mismatch. missing={missing}, unexpected={unexpected}')


def predict_probability(model, batch):
    hidden = model.encoder(batch)
    prediction = model.readout_prob(hidden).squeeze(-1)
    if model.args.learning_target == 'probability':
        return torch.clamp(prediction, 0.0, 1.0)
    pred_cov = torch.clamp(prediction, -1.0, 1.0)
    return cov2prob(batch, pred_cov)


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    device = torch.device('cpu' if args.use_cpu or not torch.cuda.is_available() else f'cuda:{args.gpu_id}')

    dataset = GraphDataset(args.data_path, is_train=False, val_list=args.val_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = TRACETrainer(args)
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    maes, all_target, all_pred = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = batch.to(device)
            pred_prob = predict_probability(model, batch)
            target = batch.prob
            mae = torch.nn.functional.l1_loss(pred_prob, target)
            maes.append(float(mae.detach().cpu()))
            all_target.append(target.detach().cpu())
            all_pred.append(pred_prob.detach().cpu())
            print(f'batch {batch_idx + 1}/{len(loader)} mae {maes[-1]:.6f}')

    all_target = torch.cat(all_target).numpy()
    all_pred = torch.cat(all_pred).numpy()
    global_mae = float(torch.nn.functional.l1_loss(torch.tensor(all_pred), torch.tensor(all_target)))
    global_r2 = float(r2_score(all_target, all_pred))
    print(f'RESULT batch_mae {sum(maes) / len(maes):.6f} global_mae {global_mae:.6f} global_r2 {global_r2:.6f}')


if __name__ == '__main__':
    main()
