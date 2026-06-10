from pathlib import Path
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset


class OrderedData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'forward_index' in key or 'backward_index' in key:
            return 0
        if 'edge_index' in key:
            return 1
        if key in ['tt_pair_index', 'connect_pair_index']:
            return 1
        return 0


def _as_tensor(value, dtype=None):
    tensor = torch.tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def parse_pm_graph(circuit):
    x = _as_tensor(circuit['x'], torch.long)
    edge_index = _as_tensor(circuit['edge_index'], torch.long).contiguous()
    graph = OrderedData(x=x, edge_index=edge_index)
    graph.gate = x[:, 1:2].float()
    graph.prob = _as_tensor(circuit['prob'], torch.float).reshape((len(x),))
    graph.forward_level = _as_tensor(circuit['forward_level'], torch.long)
    graph.forward_index = _as_tensor(circuit['forward_index'], torch.long)
    graph.backward_level = _as_tensor(circuit['backward_level'], torch.long)
    graph.backward_index = _as_tensor(circuit['backward_index'], torch.long)
    return graph


class PMDataset(InMemoryDataset):
    def __init__(self, root, npz_path, transform=None, pre_transform=None, pre_filter=None):
        self.npz_path = str(Path(npz_path).expanduser())
        super().__init__(str(Path(root).expanduser()), transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        return [self.npz_path]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        circuits = np.load(self.npz_path, allow_pickle=True)['circuits'].item()
        data_list = [parse_pm_graph(circuits[name]) for name in circuits]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_train_valid_dataset(work_dir, npz_path, trainval_split=0.9, random_shuffle=True, seed=None, sample_ratio=1.0):
    dataset = PMDataset(work_dir, npz_path)
    if random_shuffle:
        if seed is None:
            dataset = dataset[torch.randperm(len(dataset))]
        else:
            generator = torch.Generator().manual_seed(seed)
            dataset = dataset[torch.randperm(len(dataset), generator=generator)]
    cutoff = int(len(dataset) * trainval_split)
    train_dataset = dataset[:cutoff]
    valid_dataset = dataset[cutoff:]
    if sample_ratio < 1.0:
        train_dataset = train_dataset[:int(len(train_dataset) * sample_ratio)]
        valid_dataset = valid_dataset[:int(len(valid_dataset) * sample_ratio)]
    return train_dataset, valid_dataset
