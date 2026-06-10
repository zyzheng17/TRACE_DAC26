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
        self.hf_init = MLP(
            64,
            self.dim_hidden,
            self.dim_hidden,
            num_layer=3,
            p_drop=0.2,
            norm_layer='batchnorm',
            act_layer='relu',
        )
        self.final_mlp = MLP(2 * self.dim_hidden, self.dim_hidden, self.dim_hidden, 3)
        self.readout_prob = MLP(
            self.dim_hidden,
            32,
            1,
            num_layer=3,
            p_drop=0.2,
            norm_layer='batchnorm',
            act_layer='relu',
        )
        self.pos_enc = nn.Embedding(10, self.dim_hidden)

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

    def forward(self, x, edge_index, forward_level, forward_index):
        device = next(self.parameters()).device
        max_num_layers = torch.max(forward_level).item() + 1
        min_num_layers = 1

        h = self.hf_init(x.float().to(device))
        h[forward_level == 0] = 0.5

        for _ in range(self.num_rounds):
            for level in range(min_num_layers, max_num_layers):
                h_new = h.clone()
                l_and_node = forward_index[forward_level == level]

                if l_and_node.size(0) > 0:
                    and_mask = forward_level[edge_index[1]] == level
                    and_edge_index = edge_index[:, and_mask]

                    and_tgt_node_idx = and_edge_index[1]
                    and_src_node_idx = and_edge_index[0]

                    unique_tgt_nodes, inverse_indices = torch.unique(and_tgt_node_idx, return_inverse=True)
                    max_length = torch.bincount(inverse_indices).max().item()
                    index_matrix = torch.full(
                        (len(unique_tgt_nodes), max_length),
                        -1,
                        dtype=torch.long,
                        device=h.device,
                    )
                    padding_mask = torch.ones(
                        (len(unique_tgt_nodes), max_length),
                        dtype=torch.bool,
                        device=h.device,
                    )

                    src_counts = torch.bincount(inverse_indices)
                    src_offsets = torch.cumsum(src_counts, dim=0) - src_counts
                    src_positions = torch.arange(and_src_node_idx.size(0), device=h.device) - src_offsets[inverse_indices]

                    index_matrix[inverse_indices, src_positions] = and_src_node_idx
                    padding_mask[inverse_indices, src_positions] = False

                    padded_sequences = torch.cat([h[unique_tgt_nodes].unsqueeze(1), h[index_matrix]], dim=1)
                    padding_mask = torch.cat(
                        [torch.zeros((len(unique_tgt_nodes), 1), dtype=torch.bool, device=h.device), padding_mask],
                        dim=1,
                    )
                    pos = torch.arange(padded_sequences.shape[1], device=h.device).unsqueeze(0).repeat(padded_sequences.shape[0], 1)
                    aggr = self.aggr_and(padded_sequences + self.pos_enc(pos), src_key_padding_mask=padding_mask)[:, 0]
                    h_new[unique_tgt_nodes, :] = aggr

                h = h_new
            del h_new
        return h

    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        return torch.clamp(prob, min=-1.0, max=1.0)
