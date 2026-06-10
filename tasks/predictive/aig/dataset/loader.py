import numpy as np
import os
import torch
from torch_geometric.data import Data, Dataset


class OrderedData(Data):
    def __init__(self): 
        super().__init__()
    
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ninp_node_index' or key == 'ninh_node_index':
            return self.num_nodes
        elif key == 'ninp_path_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'ninh_hop_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'hop_pi' or key == 'hop_po' or key == 'hop_nodes': 
            return self.num_nodes
        elif key == 'winhop_po' or key == 'winhop_nodes':
            return self.num_nodes
        elif key == 'hop_pair_index' or key == 'hop_forward_index':
            return args[0]['hop_forward_index'].shape[0]
        elif key == 'path_forward_index':
            return args[0]['path_forward_index'].shape[0]
        elif key == 'paths' or key == 'hop_nodes':
            return self.num_nodes
        elif 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'forward_index' or key == 'backward_index':
            return 0
        elif key == "edge_index" or key == 'tt_pair_index' or key == 'rc_pair_index':
            return 1
        elif key == "connect_pair_index" or key == 'hop_pair_index':
            return 1
        else:
            return 0
        
class GraphDataset(Dataset):
    def __init__(self, data_dir, is_train, val_list):
        super(GraphDataset, self).__init__()

        data_list = self.load_graph_pt(data_dir)
        if is_train:
            data_list = [data for data in data_list if data.name not in val_list]
        else:
            data_list = [data for data in data_list if data.name in val_list]

        data_list = self.data_preprocess(data_list)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def data_preprocess(self, data_list):
        new_data_list = []
        for data in data_list:
            and_edge_index = data.edge_index[:,data.gate.squeeze()[data.edge_index[1]] == 1]
            and_tgt_node_idx = and_edge_index[1].reshape(-1,2)[:,0]
            and_src_node_idx = and_edge_index[0].reshape(-1,2)
            data.prob = torch.tensor(data.prob)
            data.cov = torch.zeros_like(data.prob)
            data.cov[and_tgt_node_idx] = data.prob[and_tgt_node_idx] - data.prob[and_src_node_idx[:,0]] * data.prob[and_src_node_idx[:,1]]
            del data.name
            new_data_list.append(data)
            
        return new_data_list

    def load_graph_pt_dir(self, dir):
        graphs = []
        for root, _, files in os.walk(dir):
            for file in files:  
                if file.endswith('.pt'):
                    file_path = os.path.join(root, file)
                    graph = torch.load(file_path,weights_only=False)
                    graphs.append(graph)
        return graphs
    
    def load_graph_pt(self, path):
        data_list = torch.load(path, weights_only=False)
        return data_list

    def load_graph_npz(self, path):
        data_list = np.load(path, allow_pickle=True)['circuits'].item()
        
        return data_list

