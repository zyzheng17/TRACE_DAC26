import torch
from torch import nn

from expr_graphs import NODE_TYPES


class TRACEComputationalGraph(nn.Module):
    """A minimal TRACE-style encoder for arithmetic computational graphs."""

    def __init__(self, p=97, hidden=128, num_layers=2, num_heads=8, max_arity=4):
        super().__init__()
        self.p = p
        self.hidden = hidden
        self.max_arity = max_arity
        self.type_embedding = nn.Embedding(len(NODE_TYPES), hidden)
        self.value_embedding = nn.Embedding(p, hidden)
        self.position_embedding = nn.Embedding(max_arity + 1, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=num_heads,
            dim_feedforward=hidden * 2,
            batch_first=True,
        )
        self.operator_encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch, return_trace=False):
        h = self.type_embedding(batch.node_type) + self.value_embedding(batch.node_value.clamp(0, self.p - 1))
        max_level = int(batch.forward_level.max().item())
        device = h.device
        forward_trace = []

        for level in range(1, max_level + 1):
            h_next = h.clone()
            target_nodes = batch.forward_index[batch.forward_level == level] if hasattr(batch, 'forward_index') else None
            if target_nodes is None:
                target_nodes = (batch.forward_level == level).nonzero(as_tuple=True)[0]
            if target_nodes.numel() == 0:
                continue

            sequence = h.new_zeros(target_nodes.numel(), self.max_arity + 1, self.hidden)
            padding_mask = torch.ones(target_nodes.numel(), self.max_arity + 1, dtype=torch.bool, device=device)
            sequence[:, 0] = h[target_nodes]
            padding_mask[:, 0] = False
            level_trace = []

            for row, target in enumerate(target_nodes.tolist()):
                edge_ids = (batch.edge_index[1] == target).nonzero(as_tuple=True)[0]
                if edge_ids.numel() == 0:
                    continue
                edge_ids = edge_ids[torch.argsort(batch.edge_pos[edge_ids])]
                source_nodes = batch.edge_index[0, edge_ids]
                arity = min(source_nodes.numel(), self.max_arity)
                source_nodes = source_nodes[:arity]
                sequence[row, 1:arity + 1] = h[source_nodes]
                padding_mask[row, 1:arity + 1] = False
                level_trace.append({
                    'target': target,
                    'sources': source_nodes.detach().cpu().tolist(),
                    'positions': batch.edge_pos[edge_ids[:arity]].detach().cpu().tolist(),
                    'sequence_nodes': [target] + source_nodes.detach().cpu().tolist(),
                })

            positions = torch.arange(self.max_arity + 1, device=device)
            encoded = self.operator_encoder(
                sequence + self.position_embedding(positions).unsqueeze(0),
                src_key_padding_mask=padding_mask,
            )
            h_next[target_nodes] = encoded[:, 0]
            h = h_next
            forward_trace.append({
                'level': level,
                'sequence_shape': tuple(sequence.shape),
                'operators': level_trace,
            })

        output = self.readout(h[self._output_indices(batch)]).squeeze(-1)
        if return_trace:
            return output, forward_trace
        return output

    @staticmethod
    def _output_indices(batch):
        output_idx = batch.output_node_idx.view(-1)
        if hasattr(batch, 'ptr') and batch.ptr is not None:
            return batch.ptr[:-1].to(output_idx.device) + output_idx
        return output_idx
