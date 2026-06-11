import torch
from torch import nn


class TRACE(nn.Module):
    def __init__(self, args, num_rounds=1, dim_hidden=128):
        super().__init__()
        self.num_rounds = num_rounds
        self.dim_hidden = dim_hidden
        self.aggr = 'transformer'
        self.args = args

        self.num_gate_types = 16
        self.aggr_modules = nn.ModuleDict(
            {f'aggr_{gate_type}': self.get_and_aggr() for gate_type in range(self.num_gate_types)}
        )
        self.hf_init = nn.Embedding(self.num_gate_types, self.dim_hidden)
        self.hf_init_gate_num = nn.Embedding(10000, self.dim_hidden)
        self.prob_embedding = nn.Embedding(101, 128)

    def get_and_aggr(self):
        tf_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_hidden,
            nhead=8,
            dim_feedforward=self.dim_hidden * 4,
            batch_first=True,
        )
        return nn.TransformerEncoder(tf_layer, num_layers=2)

    @property
    def last_shared_layer(self):
        return self.aggr_modules['aggr_0'].layers[-1].linear2

    def forward(self, G, input_pattern=None):
        max_num_layers = torch.max(G.forward_level).item() + 1
        min_num_layers = 0
        gate_type = G.gate.squeeze()

        h = self.hf_init(gate_type)

        one_hot = torch.eye(self.dim_hidden, device=G.x.device)
        pi_indices = torch.where(G.forward_level == 0)[0][:self.dim_hidden]
        if len(pi_indices) > 0:
            h[pi_indices] = one_hot[:len(pi_indices)]

        for _ in range(self.num_rounds):
            h_new = h.clone()
            for level in range(min_num_layers, max_num_layers):
                for current_gate_type in range(self.num_gate_types):
                    gate_level_mask = torch.logical_and(
                        G.gate.squeeze() == current_gate_type,
                        G.forward_level == level,
                    )
                    if not gate_level_mask.any():
                        continue

                    l_gate_nodes = G.forward_index[gate_level_mask]
                    if l_gate_nodes.size(0) == 0 or G.edge_index.size(1) == 0:
                        continue

                    edge_src_mask = G.edge_index[0] < len(G.gate)
                    edge_valid = edge_src_mask & (G.edge_index[1] < len(G.gate))
                    valid_edge_index = G.edge_index[:, edge_valid]
                    if valid_edge_index.size(1) == 0:
                        continue

                    gate_mask = torch.logical_and(
                        G.gate[valid_edge_index[0]].squeeze() == current_gate_type,
                        G.forward_level[valid_edge_index[0]] == level,
                    )
                    gate_edge_index = valid_edge_index[:, gate_mask]
                    if gate_edge_index.size(1) == 0:
                        continue

                    aggr_module = self.aggr_modules[f'aggr_{current_gate_type}']

                    if current_gate_type in [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                        if gate_edge_index.size(0) < 3:
                            continue
                        tgt_node_idx = gate_edge_index[0]
                        src1_node_idx = gate_edge_index[1]
                        src2_node_idx = gate_edge_index[2]

                        max_idx = len(h) - 1
                        tgt_node_idx = torch.clamp(tgt_node_idx, 0, max_idx)
                        src1_node_idx = torch.clamp(src1_node_idx, 0, max_idx)
                        src2_node_idx = torch.clamp(src2_node_idx, 0, max_idx)

                        aggr = torch.stack(
                            [h[tgt_node_idx], h[src1_node_idx], h[src2_node_idx]],
                            dim=1,
                        )
                        h_new[tgt_node_idx, :] = aggr_module(aggr)[:, 0]

                    elif current_gate_type == 5:
                        if gate_edge_index.size(0) < 2:
                            continue
                        tgt_node_idx = gate_edge_index[0]
                        src_node_idx = gate_edge_index[1]

                        max_idx = len(h) - 1
                        tgt_node_idx = torch.clamp(tgt_node_idx, 0, max_idx)
                        src_node_idx = torch.clamp(src_node_idx, 0, max_idx)

                        aggr = torch.stack([h[tgt_node_idx], h[src_node_idx]], dim=1)
                        h_new[tgt_node_idx, :] = aggr_module(aggr)[:, 0]

                h = h_new
            del h_new
        return h
