import os
import pickle
import sys
from pathlib import Path

import dgl
import torch
import torch.nn.functional as F
import torch_geometric.data
from dgl.dataloading import GraphDataLoader
from torch.nn.utils.rnn import pad_sequence


class OrderedData(torch_geometric.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'forward_index']:
            return self.forward_index.shape[0]
        if key in ['syn_edge_index', 'syn_forward_index']:
            return self.syn_forward_index.shape[0]
        if 'batch' in key:
            return 1
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            return 1
        return 0


DEFAULT_DATA_ROOT = Path(
    os.environ.get('TRACE_RTL_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data')
).expanduser()

# Older pickles can point at the pre-cleanup module name.
sys.modules.setdefault('dataset.rtl_parser', sys.modules[__name__])


class MyDataset(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name='my_dataset')
        self.graphs = []
        self.label = []

    def add_graph_data(self, dgl_graph, label_data):
        self.graphs.append(dgl_graph)
        self.label.append(label_data)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    def collate(self, samples):
        graphs = list(samples)
        num_nodes = [graph.num_nodes() for graph in graphs]
        max_num_nodes = max(num_nodes)
        num_graphs = len(graphs)

        attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
        node_feat, in_degree, out_degree, path_data = [], [], [], []
        dist = -torch.ones((num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long)

        dg5_gate, dg5_edge_index, dg5_forward_level, dg5_forward_index = [], [], [], []
        dg5_x, dg5_prob, dg5_num_nodes = [], [], []

        for i, graph in enumerate(graphs):
            attn_mask[i, :, num_nodes[i] + 1:] = 1
            node_feat.append(graph.ndata['feat'] + 1)
            in_degree.append(torch.clamp(graph.in_degrees() + 1, min=0, max=512))
            out_degree.append(torch.clamp(graph.out_degrees() + 1, min=0, max=512))

            path = graph.ndata['path']
            max_len = 5
            if path.size(dim=2) >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                shortest_path = F.pad(path, (0, max_len - path.size(dim=2)), 'constant', -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            shortest_path = F.pad(shortest_path, (0, 0, 0, pad_num_nodes, 0, pad_num_nodes), 'constant', -1)

            edata = torch.cat((graph.edata['feat'] + 1, torch.zeros(1, graph.edata['feat'].shape[1])), dim=0)
            path_data.append(edata[shortest_path])
            dist[i, :num_nodes[i], :num_nodes[i]] = graph.ndata['spd']

            if hasattr(graph, 'dg5_gate'):
                dg5_gate.append(graph.dg5_gate)
                dg5_edge_index.append(graph.dg5_edge_index)
                dg5_forward_level.append(graph.dg5_forward_level)
                dg5_forward_index.append(graph.dg5_forward_index)
                dg5_x.append(graph.dg5_x)
                dg5_prob.append(graph.dg5_prob)
                dg5_num_nodes.append(graph.dg5_num_nodes)

        batch = BatchWithDG5(
            attn_mask,
            pad_sequence(node_feat, batch_first=True),
            pad_sequence(in_degree, batch_first=True),
            pad_sequence(out_degree, batch_first=True),
            torch.stack(path_data),
            dist,
        )
        if dg5_gate:
            batch.dg5_gate = dg5_gate
            batch.dg5_edge_index = dg5_edge_index
            batch.dg5_forward_level = dg5_forward_level
            batch.dg5_forward_index = dg5_forward_index
            batch.dg5_x = dg5_x
            batch.dg5_prob = dg5_prob
            batch.dg5_num_nodes = dg5_num_nodes
        return batch


class BatchWithDG5:
    def __init__(self, attn_mask, node_feat, in_degree, out_degree, path_data, dist):
        self.attn_mask = attn_mask
        self.node_feat = node_feat
        self.in_degree = in_degree
        self.out_degree = out_degree
        self.path_data = path_data
        self.dist = dist

    def __iter__(self):
        return iter((self.attn_mask, self.node_feat, self.in_degree, self.out_degree, self.path_data, self.dist))

    def __len__(self):
        return 6


def _resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _rtl_pair_dir(data_root=None):
    root = _resolve_path(data_root or DEFAULT_DATA_ROOT)
    if (root / 'dataset_train_ori.pkl').exists() or (root / 'dataset_valid_ori.pkl').exists():
        return root
    return root / 'dataset_graph' / 'data_bench'


def _graph_loader(path, batch_size):
    with open(path, 'rb') as file_obj:
        dataset = pickle.load(file_obj)
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=dataset.collate,
    )


def load_train_valid_dataset_stage_align(batch_size, train_valid='train', data_root=None):
    dataset_dir = _rtl_pair_dir(data_root)
    ori_loader = _graph_loader(dataset_dir / f'dataset_{train_valid}_ori.pkl', batch_size)
    pos_loader = _graph_loader(dataset_dir / f'dataset_{train_valid}_pos.pkl', batch_size)
    return ori_loader, pos_loader
