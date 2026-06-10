import torch
from torch import nn
from .mlp import MLP


class TRACE(nn.Module):
    def __init__(self, args, num_rounds=1, dim_hidden=128):
        super(TRACE, self).__init__()

        self.num_rounds = num_rounds
        self.dim_hidden = dim_hidden
        self.aggr = 'transformer'
        self.args = args

        self.aggr_and = self.get_and_aggr()
        self.aggr_not = self.get_and_aggr()

        self.hf_init = nn.Embedding(4, self.dim_hidden)
        self.hadamard_mlp = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, 3)

        self.final_mlp = MLP(2*self.dim_hidden, self.dim_hidden, self.dim_hidden, 3)
        
        self.prob_embedding = nn.Embedding(101, 128)
        
            
    def get_and_aggr(self):
        tf_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_hidden,
            nhead=8,
            dim_feedforward=self.dim_hidden * 2,
            batch_first=True,
        )
        return nn.TransformerEncoder(tf_layer, num_layers=2)
        
    @property
    def last_shared_layer(self):
        return self.aggr_and.layers[-1].linear2

    def forward(self, gate, edge_index, forward_level, forward_index,):
        
        max_num_layers = torch.max(forward_level).item() + 1
        min_num_layers = 0
        gate_type = gate.squeeze()
        device = gate.device

        h = self.hf_init(gate_type)

        h[forward_level==0] = torch.zeros(self.dim_hidden).to(device) + 0.5

        for _ in range(self.num_rounds):
            h_new = h.clone()
            for level in range(min_num_layers, max_num_layers):
                l_and_node =forward_index[torch.logical_and(gate.squeeze() == 1, forward_level == level)]

                if l_and_node.size(0) > 0:
                    and_mask  = torch.logical_and(gate[edge_index[1]].squeeze() == 1, forward_level[edge_index[1]] == level)
                    and_edge_index = edge_index[:,and_mask]

                    and_tgt_node_idx = and_edge_index[1].reshape(-1,2)[:,0]
                    and_src_node_idx = and_edge_index[0].reshape(-1,2)

                    h_src1 = h[and_src_node_idx[:, 0]]
                    h_src2 = h[and_src_node_idx[:, 1]]
                    h_tgt = h[and_tgt_node_idx]
                    aggr = torch.stack([h_tgt, h_src1, h_src2], dim=1)
                    aggr = self.aggr_and(aggr)[:, 0]

                    h_new[and_tgt_node_idx, :] = aggr

                l_not_node =forward_index[torch.logical_and(gate.squeeze() == 2, forward_level == level)]
                if l_not_node.size(0) > 0:
                    not_mask = torch.logical_and(gate[edge_index[1]].squeeze() == 2, forward_level[edge_index[1]] == level)
                    not_edge_index = edge_index[:,not_mask]

                    not_src_node_idx = not_edge_index[0]
                    not_tgt_node_idx = not_edge_index[1]

                    h_src = h[not_src_node_idx] 
                    h_tgt = h[not_tgt_node_idx]
                    aggr = torch.stack([h_tgt, h_src], dim=1)
                    aggr = self.aggr_not(aggr)[:,0]    # 不share参数

                    h_new[not_tgt_node_idx] = aggr
                h = h_new
            del h_new
        return h