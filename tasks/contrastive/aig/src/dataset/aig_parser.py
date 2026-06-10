import sys
import types

import torch
import torch_geometric
from torch_geometric.data import Dataset

from utils.data_utils import OrderedData  # noqa: F401 - required by torch.load pickles


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
