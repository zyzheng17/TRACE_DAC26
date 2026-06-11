import torch


class TRACEGraph:
    def __init__(self, gate, edge_index, forward_level, forward_index, x, num_nodes):
        self.gate = gate
        self.edge_index = edge_index
        self.forward_level = forward_level
        self.forward_index = forward_index
        self.x = x
        self.num_nodes = int(num_nodes) if not isinstance(num_nodes, int) else num_nodes
        self.edge_weight = None
        self.num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
        self.batch = torch.zeros(self.num_nodes, dtype=torch.long, device=gate.device)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class PairedDataLoader:
    def __init__(self, orig_loader, pos_loader, encoder_type='TRACE'):
        if encoder_type != 'TRACE':
            raise ValueError('PairedDataLoader only supports TRACE batches.')
        self.orig_loader = orig_loader
        self.pos_loader = pos_loader

    def __iter__(self):
        self.orig_iter = iter(self.orig_loader)
        self.pos_iter = iter(self.pos_loader)
        return self

    def __next__(self):
        while True:
            orig_batch = next(self.orig_iter)
            pos_batch = next(self.pos_iter)
            batch = self._create_batch(orig_batch, pos_batch)
            if batch is not None:
                return batch

    def _create_batch(self, orig_batch, pos_batch):
        if not hasattr(orig_batch, 'dg5_gate') or not hasattr(pos_batch, 'dg5_gate'):
            return None

        orig_graphs, syn_graphs = [], []
        for data in zip(
            orig_batch.dg5_gate,
            orig_batch.dg5_edge_index,
            orig_batch.dg5_forward_level,
            orig_batch.dg5_forward_index,
            orig_batch.dg5_x,
            orig_batch.dg5_num_nodes,
            pos_batch.dg5_gate,
            pos_batch.dg5_edge_index,
            pos_batch.dg5_forward_level,
            pos_batch.dg5_forward_index,
            pos_batch.dg5_x,
            pos_batch.dg5_num_nodes,
        ):
            orig_graphs.append(TRACEGraph(*data[:6]))
            syn_graphs.append(TRACEGraph(*data[6:]))

        class Batch:
            pass

        batch = Batch()
        batch.gate = orig_graphs
        batch.syn_gate = syn_graphs
        batch.batch_size = len(orig_graphs)
        return batch

    def __len__(self):
        return min(len(self.orig_loader), len(self.pos_loader))
