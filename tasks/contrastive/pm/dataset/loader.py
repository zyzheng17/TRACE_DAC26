import sys
import types

import torch
import torch_geometric
from torch_geometric.data import Dataset

class OrderedData(torch_geometric.data.Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ['edge_index', 'forward_index']:
            return self.forward_index.shape[0]
        if key in ['rd_edge_index', 'rd_forward_index']:
            return self.rd_forward_index.shape[0]
        if key in ['syn_edge_index', 'syn_forward_index']:
            return self.syn_forward_index.shape[0]
        if key in ['pm_edge_index', 'pm_forward_index']:
            return self.pm_forward_index.shape[0]
        if 'batch' in key:
            return 1
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'edge_index' in key:
            return 1
        return 0


legacy_module = types.ModuleType('gen_pkl_for_subgraph_mining')
legacy_module.OrderedData = OrderedData
sys.modules.setdefault('gen_pkl_for_subgraph_mining', legacy_module)


class GraphDataset(Dataset):
    def __init__(self, data_path, size_thre=20):
        super().__init__()
        data_list = torch.load(data_path, weights_only=False)
        if size_thre is not None:
            data_list = [graph for graph in data_list if graph.pm_edge_index.shape[1] >= size_thre]

        filtered = []
        for graph in data_list:
            has_one_pm_output = torch.logical_and(graph.pm_forward_level != 0, graph.pm_backward_level == 0).sum() == 1
            has_one_syn_output = torch.logical_and(graph.syn_forward_level != 0, graph.syn_backward_level == 0).sum() == 1
            if has_one_pm_output and has_one_syn_output:
                graph.pm_edge_index = torch_geometric.utils.sort_edge_index(graph.pm_edge_index.flip(0)).flip(0)
                graph.syn_edge_index = torch_geometric.utils.sort_edge_index(graph.syn_edge_index.flip(0)).flip(0)
                filtered.append(graph)
        self.data_list = filtered

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
