from dgl.dataloading import GraphDataLoader
from pathlib import Path
import os
import pickle, json
import torch
import torch.nn.functional as F
import dgl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


DEFAULT_DATA_ROOT = Path(
    os.environ.get('TRACE_RTL_DATA_ROOT', Path(__file__).resolve().parents[2] / 'data')
).expanduser()


def _resolve_path(path):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _dataset_root(data_root=None):
    return _resolve_path(data_root or DEFAULT_DATA_ROOT)


def _rtl_pair_dir(data_root=None):
    root = _dataset_root(data_root)
    if (root / 'dataset_train_ori.pkl').exists() or (root / 'dataset_valid_ori.pkl').exists():
        return root
    return root / 'dataset_graph' / 'data_bench'

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

    def convert(self):
        self.label = torch.FloatTensor(self.label)

    def to_device(self, device):
        for i in range(len(self.graphs)):
            self.graphs[i] = self.graphs[i].to(device)
        # self.label = self.label.to(device)

    def collate(self, samples):
        # graphs, _ = map(list, zip(*samples))
        # labels = torch.stack(labels)
        graphs = list(samples)

        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        # Graphormer adds a virual node to the graph, which is connected to
        # all other nodes and supposed to represent the graph embedding. So
        # here +1 is for the virtual node.
        attn_mask = torch.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        # Since shortest_dist returns -1 for unreachable node pairs and padded
        # nodes are unreachable to others, distance relevant to padded nodes
        # use -1 padding as well.
        dist = -torch.ones(
            (num_graphs, max_num_nodes, max_num_nodes), dtype=torch.long
        )

        # DG5数据字段
        dg5_gate = []
        dg5_edge_index = []
        dg5_forward_level = []
        dg5_forward_index = []
        dg5_x = []
        dg5_prob = []
        dg5_num_nodes = []

        for i in range(num_graphs):
            # A binary mask where invalid positions are indicated by True.
            attn_mask[i, :, num_nodes[i] + 1 :] = 1

            # +1 to distinguish padded non-existing nodes from real nodes
            node_feat.append(graphs[i].ndata["feat"] + 1)

            in_degree.append(
                torch.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
            )
            out_degree.append(
                torch.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
            )

            # Path padding to make all paths to the same length "max_len".
            path = graphs[i].ndata["path"]
            path_len = path.size(dim=2)
            # shape of shortest_path: [n, n, max_len]
            max_len = 5
            if path_len >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                p1d = (0, max_len - path_len)
                # Use the same -1 padding as shortest_dist for
                # invalid edge IDs.
                shortest_path = F.pad(path, p1d, "constant", -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)
            # +1 to distinguish padded non-existing edges from real edges
            edata = graphs[i].edata["feat"] + 1
            # shortest_dist pads non-existing edges (at the end of shortest
            # paths) with edge IDs -1, and torch.zeros(1, edata.shape[1]) stands
            # for all padded edge features.
            edata = torch.cat(
                (edata, torch.zeros(1, edata.shape[1]).to(edata.device)), dim=0
            )
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

            # 提取DG5数据（如果存在）
            if hasattr(graphs[i], 'dg5_gate'):
                dg5_gate.append(graphs[i].dg5_gate)
                dg5_edge_index.append(graphs[i].dg5_edge_index)
                dg5_forward_level.append(graphs[i].dg5_forward_level)
                dg5_forward_index.append(graphs[i].dg5_forward_index)
                dg5_x.append(graphs[i].dg5_x)
                dg5_prob.append(graphs[i].dg5_prob)
                dg5_num_nodes.append(graphs[i].dg5_num_nodes)

        # node feat padding
        node_feat = pad_sequence(node_feat, batch_first=True)

        # degree padding
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)
        del graphs, samples, edata

        # 创建返回的batch对象
        batch = (
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            torch.stack(path_data),
            dist
        )
        
        # 如果存在DG5数据，添加到batch对象中
        if dg5_gate:
            # 创建dg5_data列表
            dg5_data = [
                dg5_gate[0],  # 使用第一个图的DG5数据
                dg5_edge_index[0],
                dg5_forward_level[0],
                dg5_forward_index[0],
                dg5_x[0],
                dg5_prob[0],
                dg5_num_nodes[0]
            ]
            
            # 创建带有DG5属性的batch对象
            class BatchWithDG5:
                def __init__(self, batch_data, dg5_data):
                    self.attn_mask = batch_data[0]
                    self.node_feat = batch_data[1]
                    self.in_degree = batch_data[2]
                    self.out_degree = batch_data[3]
                    self.path_data = batch_data[4]
                    self.dist = batch_data[5]
                    
                    # DG5数据
                    self.dg5_gate = dg5_data[0]
                    self.dg5_edge_index = dg5_data[1]
                    self.dg5_forward_level = dg5_data[2]
                    self.dg5_forward_index = dg5_data[3]
                    self.dg5_x = dg5_data[4]
                    self.dg5_prob = dg5_data[5]
                    self.dg5_num_nodes = dg5_data[6]
                
                def __getitem__(self, idx):
                    if idx == 0:
                        return self.attn_mask
                    elif idx == 1:
                        return self.node_feat
                    elif idx == 2:
                        return self.in_degree
                    elif idx == 3:
                        return self.out_degree
                    elif idx == 4:
                        return self.path_data
                    elif idx == 5:
                        return self.dist
                    else:
                        raise IndexError("Index out of range")
                
                def __len__(self):
                    return 6
            
            return BatchWithDG5(batch, dg5_data)
        
        return batch


class TextEmbedDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.embed = []
        self.label = []
        
    def add_text_embed_data(self, embed_dict, label_data):
        self.embed.append(embed_dict)
        self.label.append(label_data)
    
    def __getitem__(self, idx):
        return self.embed[idx]
    
    def __len__(self):
        return len(self.embed)

    def collate(self, samples):
        embed= list(samples)
        outputs, embeddings = [], []

        for i in range(len(embed)):
            outputs.append(embed[i]['outputs'])
            embeddings.append(embed[i]['embeddings'])

        # Determine the maximum size along dimension 1
        max_output_size_1 = max([output.size(1) for output in outputs])
        max_output_size_2 = max([output.size(2) for output in outputs])

        # Pad each tensor to the maximum size
        outputs = [F.pad(output, (0, max_output_size_2 - output.size(2), 0, max_output_size_1 - output.size(1)), "constant", 0) for output in outputs]

        # list to tensor
        outputs = torch.stack(outputs).squeeze(1)


        ### stack embeddings
        embeddings = torch.stack(embeddings)

        return (
            (outputs, embeddings)
        )


class RegGraphDataset(dgl.data.DGLDataset):
    def __init__(self):
        super().__init__(name='my_dataset')
        self.graphs = []
        self.label = []
        
    def add_graph_data(self, dgl_graph, label_data):
        self.graphs.append(dgl_graph)
        self.label.append(label_data)
    
    def __getitem__(self, idx):
        return (self.graphs[idx], self.label[idx])
    
    def __len__(self):
        return len(self.graphs)


class NetDataset(dgl.data.DGLDataset):
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

def load_train_valid_dataset(batch_size, train_valid="train"):
    shuffle_tf = False

    dataset_dir = f"../../dataset/"

    print(f"Loading graph dataset ...")
    #### load graph train data ####
    with open(f'{dataset_dir}/dataset_graph/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
        graph_train_ori = pickle.load(f)
    graph_ori_loader = GraphDataLoader(
        graph_train_ori,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        collate_fn=graph_train_ori.collate
    )
    del graph_train_ori

    with open(f'{dataset_dir}/dataset_graph/data_bench/dataset_{train_valid}_pos.pkl', 'rb') as f:
        graph_train_pos = pickle.load(f)
    graph_pos_loader = GraphDataLoader(
        graph_train_pos,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        collate_fn=graph_train_pos.collate
    )
    del graph_train_pos

    # with open(f'{dataset_dir}/dataset_graph/data_bench/dataset_{train_valid}_neg.pkl', 'rb') as f:
    #     graph_train_neg = pickle.load(f)
    # graph_neg_loader = GraphDataLoader(
    #     graph_train_neg,
    #     batch_size=batch_size,
    #     shuffle=shuffle_tf,
    #     collate_fn=graph_train_neg.collate
    # )
    # del graph_train_neg

    # ### load rtl text data ###
    # print(f"Loading text dataset ...")
    # with open(f'{dataset_dir}/dataset_context/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
    #     text_data_ori = pickle.load(f)
    # text_loader = DataLoader(
    #     text_data_ori,
    #     batch_size=batch_size,
    #     shuffle=shuffle_tf,
    #     collate_fn=text_data_ori.collate
    # )
    # del text_data_ori


    # ### load text summary data ###
    # print(f"Loading summary dataset ...")
    # with open(f"{dataset_dir}/dataset_summary/{train_valid}.json", 'r') as f:
    #     summary_train_data = json.load(f)
    # summary_loader = DataLoader(
    #     summary_train_data, 
    #     batch_size=batch_size, 
    #     shuffle=shuffle_tf
    # )
    # del summary_train_data 

    # train_loader_rtl = (graph_ori_loader, graph_pos_loader, graph_neg_loader, summary_loader, text_loader)

    train_loader_rtl = (graph_ori_loader, graph_pos_loader)

    return train_loader_rtl


def load_test_dataset(design, batch_size=32):
    shuffle_tf = False

    dataset_dir = f"../../dataset/"

    #### load graph train data ####
    with open(f'{dataset_dir}/dataset_graph/data_bench/{design}.pkl', 'rb') as f:
        graph_data = pickle.load(f)
    graph_loader = GraphDataLoader(
        graph_data,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        collate_fn=graph_data.collate
    )
    del graph_data

    ### load rtl text data ###
    with open(f'{dataset_dir}/dataset_context/data_bench/{design}.pkl', 'rb') as f:
        text_data_ori = pickle.load(f)
    text_loader = DataLoader(
        text_data_ori,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        collate_fn=text_data_ori.collate
    )
    del text_data_ori


    ### load text summary data ###
    with open(f"{dataset_dir}/dataset_summary/{design}.json", 'r') as f:
        summary_data = json.load(f)
    summary_loader = DataLoader(
        summary_data, 
        batch_size=batch_size, 
        shuffle=shuffle_tf
    )
    del summary_data

    test_loader = (graph_loader, summary_loader, text_loader)

    return test_loader


def load_finetune_dataset(design, batch_size=32):
    shuffle_tf = False
    dataset_dir = f"../../dataset/"

    with open(f'{dataset_dir}/dataset_finetune/data_bench/{design}.pkl', 'rb') as f:
        graph_data = pickle.load(f)
    reg_graph_loader = GraphDataLoader(
        graph_data,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del graph_data

    return reg_graph_loader


def load_train_valid_dataset_net(batch_size, train_valid="train"):
    shuffle_tf = False

    dataset_dir = f"../../dataset/"

    print(f"Loading graph dataset ...")
    #### load graph train data ####
    with open(f'{dataset_dir}/dataset_net/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
        net_train_ori = pickle.load(f)
    net_ori_loader = GraphDataLoader(
        net_train_ori,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_train_ori

    with open(f'{dataset_dir}/dataset_net/data_bench/dataset_{train_valid}_pos.pkl', 'rb') as f:
        net_train_pos = pickle.load(f)
    net_pos_loader = GraphDataLoader(
        net_train_pos,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_train_pos

    with open(f'{dataset_dir}/dataset_net/data_bench/dataset_{train_valid}_neg.pkl', 'rb') as f:
        net_train_neg = pickle.load(f)
    net_neg_loader = GraphDataLoader(
        net_train_neg,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_train_neg

    train_loader_net = (net_ori_loader, net_pos_loader, net_neg_loader)

    return train_loader_net

def load_train_valid_dataset_stage_align(batch_size, train_valid="train", data_root=None):
    shuffle_tf = False

    dataset_dir = _rtl_pair_dir(data_root)

    print(f"Loading RTL dataset ...")
    with open(dataset_dir / f'dataset_{train_valid}_ori.pkl', 'rb') as f:
        graph_train_ori = pickle.load(f)
    graph_ori_loader = GraphDataLoader(
        graph_train_ori,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        drop_last=True,
        collate_fn=graph_train_ori.collate
    )
    del graph_train_ori

    with open(dataset_dir / f'dataset_{train_valid}_pos.pkl', 'rb') as f:
        graph_train_pos = pickle.load(f)
    graph_pos_loader = GraphDataLoader(
        graph_train_pos,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        drop_last=True,
        collate_fn=graph_train_pos.collate
    )
    del graph_train_pos

    # with open(f'{dataset_dir}/dataset_graph/data_bench/dataset_{train_valid}_neg.pkl', 'rb') as f:
    #     graph_train_neg = pickle.load(f)
    # graph_neg_loader = GraphDataLoader(
    #     graph_train_neg,
    #     batch_size=batch_size,
    #     shuffle=shuffle_tf,
    #     collate_fn=graph_train_neg.collate
    # )
    # del graph_train_neg

    # ### load rtl text data ###
    # print(f"Loading text dataset ...")
    # with open(f'{dataset_dir}/dataset_context/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
    #     text_data_ori = pickle.load(f)
    # text_loader_ori = DataLoader(
    #     text_data_ori,
    #     batch_size=batch_size,
    #     shuffle=shuffle_tf,
    #     collate_fn=text_data_ori.collate
    # )
    # text_loader_neg = DataLoader(
    #     text_data_ori,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=text_data_ori.collate
    # )
    # del text_data_ori


    # ### load text summary data ###
    # print(f"Loading summary dataset ...")
    # with open(f"{dataset_dir}/dataset_summary/{train_valid}.json", 'r') as f:
    #     summary_train_data = json.load(f)
    # summary_loader = DataLoader(
    #     summary_train_data, 
    #     batch_size=batch_size, 
    #     shuffle=shuffle_tf
    # )
    # del summary_train_data 



    # print(f"Loading netlist dataset ...") 
    # #### load graph train data ####
    # with open(f'{dataset_dir}/dataset_net/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
    #     net_train_ori = pickle.load(f)
    # net_ori_loader = GraphDataLoader(
    #     net_train_ori,
    #     batch_size=batch_size,
    #     shuffle=shuffle_tf,
    # )
    # del net_train_ori

    # with open(f'{dataset_dir}/dataset_net/data_bench/dataset_{train_valid}_neg.pkl', 'rb') as f:
    #     net_train_neg = pickle.load(f)
    # net_neg_loader = GraphDataLoader(
    #     net_train_neg,
    #     batch_size=batch_size,
    #     shuffle=shuffle_tf,
    # )
    # del net_train_neg


    # loader_align = (graph_ori_loader, graph_pos_loader, graph_neg_loader,\
    #                 summary_loader, text_loader_ori, text_loader_neg,\
    #                 net_ori_loader, net_neg_loader)
    loader_align = (graph_ori_loader, graph_pos_loader)
    

    return loader_align


def bak_load_train_valid_dataset_stage_align(batch_size, train_valid="train"):
    shuffle_tf = False

    dataset_dir = f"../../dataset/"

    print(f"Loading RTL dataset ...")
    with open(f'{dataset_dir}/dataset_graph/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
        graph_train_ori = pickle.load(f)
    graph_ori_loader = GraphDataLoader(
        graph_train_ori,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        collate_fn=graph_train_ori.collate
    )
    del graph_train_ori

    ### load rtl text data ###
    print(f"Loading text dataset ...")
    with open(f'{dataset_dir}/dataset_context/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
        text_data_ori = pickle.load(f)
    text_loader_ori = DataLoader(
        text_data_ori,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        collate_fn=text_data_ori.collate
    )
    del text_data_ori

    with open(f'{dataset_dir}/dataset_context/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
        text_data_neg = pickle.load(f)
    text_loader_neg = DataLoader(
        text_data_ori,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=text_data_ori.collate
    )
    del text_data_neg


    ### load text summary data ###
    print(f"Loading summary dataset ...")
    with open(f"{dataset_dir}/dataset_summary/{train_valid}.json", 'r') as f:
        summary_train_data = json.load(f)
    summary_loader = DataLoader(
        summary_train_data, 
        batch_size=batch_size, 
        shuffle=shuffle_tf
    )
    del summary_train_data 



    print(f"Loading netlist dataset ...") 
    #### load graph train data ####
    with open(f'{dataset_dir}/dataset_net/data_bench/dataset_{train_valid}_ori.pkl', 'rb') as f:
        net_train_ori = pickle.load(f)
    net_ori_loader = GraphDataLoader(
        net_train_ori,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_train_ori

    with open(f'{dataset_dir}/dataset_net/data_bench/dataset_{train_valid}_neg.pkl', 'rb') as f:
        net_train_neg = pickle.load(f)
    net_neg_loader = GraphDataLoader(
        net_train_neg,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_train_neg


    loader_align = (graph_ori_loader, summary_loader, text_loader_ori,\
                    net_ori_loader, net_neg_loader)

    return loader_align