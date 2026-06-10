import torch
from torch import nn

from .mlp import MLP


class TRACE(nn.Module):
    def __init__(self, args, num_rounds=1, dim_hidden=128):
        super().__init__()
        self.num_rounds = num_rounds
        self.dim_hidden = dim_hidden
        self.aggr = 'transformer'
        self.args = args

        self.aggr_and = self.get_and_aggr()
        self.aggr_not = MLP(self.dim_hidden, 2 * self.dim_hidden, self.dim_hidden, 5)
        self.hf_init = nn.Embedding(4, self.dim_hidden)

    def get_and_aggr(self):
        tf_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_hidden,
            nhead=8,
            dim_feedforward=self.dim_hidden * 2,
            batch_first=True,
        )
        return nn.TransformerEncoder(tf_layer, num_layers=5)

    def forward(self, G):
        max_num_layers = torch.max(G.forward_level).item() + 1
        min_num_layers = 0
        gate_type = G.gate.squeeze()

        h = self.hf_init(gate_type)
        h[G.forward_level == 0] = G.prob[G.forward_level == 0].unsqueeze(-1).repeat(1, self.dim_hidden)

        for _ in range(self.num_rounds):
            h_new = h.clone()
            for level in range(min_num_layers, max_num_layers):
                l_and_node = G.forward_index[torch.logical_and(G.gate.squeeze() == 1, G.forward_level == level)]

                if l_and_node.size(0) > 0:
                    and_mask = torch.logical_and(
                        G.gate[G.edge_index[1]].squeeze() == 1,
                        G.forward_level[G.edge_index[1]] == level,
                    )
                    and_edge_index = G.edge_index[:, and_mask]

                    and_tgt_node_idx = and_edge_index[1].reshape(-1, 2)[:, 0]
                    and_src_node_idx = and_edge_index[0].reshape(-1, 2)

                    aggr = torch.stack(
                        [h[and_tgt_node_idx], h[and_src_node_idx[:, 0]], h[and_src_node_idx[:, 1]]],
                        dim=1,
                    )
                    aggr = self.aggr_and(aggr)[:, 0]
                    h_new[and_tgt_node_idx, :] = aggr

                l_not_node = G.forward_index[torch.logical_and(G.gate.squeeze() == 2, G.forward_level == level)]
                if l_not_node.size(0) > 0:
                    not_mask = torch.logical_and(
                        G.gate[G.edge_index[1]].squeeze() == 2,
                        G.forward_level[G.edge_index[1]] == level,
                    )
                    not_edge_index = G.edge_index[:, not_mask]

                    not_src_node_idx = not_edge_index[0]
                    not_tgt_node_idx = not_edge_index[1]

                    aggr = self.aggr_not(h[not_src_node_idx])
                    h_new[not_tgt_node_idx] = aggr
                h = h_new
        return h
