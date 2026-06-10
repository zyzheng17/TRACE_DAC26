import argparse
import random
from pathlib import Path

import torch


def load_split(input_folder: Path, split_file: Path | None, seed: int, ratio: float) -> set[str]:
    pt_files = sorted(input_folder.glob('*.pt'))
    circuit_names = sorted({'_'.join(path.stem.split('_')[:-4]) for path in pt_files})
    if split_file and split_file.exists():
        return {line.strip() for line in split_file.read_text().splitlines() if line.strip()}
    rng = random.Random(seed)
    val_names = set(rng.sample(circuit_names, int(len(circuit_names) * ratio))) if circuit_names else set()
    if split_file:
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text('\n'.join(sorted(val_names)) + ('\n' if val_names else ''))
    return val_names


def merge_pt_files(input_folder: Path, output_prefix: Path, split_file: Path | None, seed: int, val_ratio: float) -> None:
    pt_files = sorted(input_folder.glob('*.pt'))
    val_names = load_split(input_folder, split_file, seed, val_ratio)
    train_data, test_data = [], []
    for idx, path in enumerate(pt_files):
        if idx % 100 == 0:
            print(f'process {idx}/{len(pt_files)}')
        graph = torch.load(path, weights_only=False)
        circuit_name = '_'.join(path.stem.split('_')[:-4])
        (test_data if circuit_name in val_names else train_data).append(graph)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save(test_data, str(output_prefix) + '_test.pt')
    torch.save(train_data, str(output_prefix) + '_train.pt')
    print(f'saved {len(train_data)} train and {len(test_data)} test graphs with prefix {output_prefix}')


def parse_args():
    parser = argparse.ArgumentParser(description='Merge per-circuit PM graph .pt files into TRACE train/test files.')
    parser.add_argument('--input_folder', type=Path, required=True)
    parser.add_argument('--output_prefix', type=Path, required=True)
    parser.add_argument('--split_file', type=Path, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    merge_pt_files(args.input_folder, args.output_prefix, args.split_file, args.seed, args.val_ratio)
