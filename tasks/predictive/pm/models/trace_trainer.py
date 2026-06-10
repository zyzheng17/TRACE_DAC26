import torch
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import r2_score


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_pred, num_layer=3, p_drop=0.2):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(dim_in, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ReLU(inplace=True), nn.Dropout(p_drop)])
        for _ in range(num_layer - 2):
            layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.BatchNorm1d(dim_hidden), nn.ReLU(inplace=True), nn.Dropout(p_drop)])
        layers.append(nn.Linear(dim_hidden, dim_pred))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class PlainMLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_pred, num_layer=3, p_drop=0.5):
        super().__init__()
        layers = [nn.Linear(dim_in, dim_hidden), nn.Dropout(p_drop)]
        for _ in range(num_layer - 2):
            layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.Dropout(p_drop)])
        layers.append(nn.Linear(dim_hidden, dim_pred))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class TRACE(nn.Module):
    """TRACE encoder for PM netlist predictive learning."""

    def __init__(self, num_rounds=1, dim_hidden=128, num_heads=8, num_layers=2):
        super().__init__()
        self.num_rounds = num_rounds
        self.dim_hidden = dim_hidden
        self.hf_init = MLP(64, dim_hidden, dim_hidden, num_layer=3, p_drop=0.2)
        self.level_embedding = nn.Embedding(300, dim_hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=dim_hidden * 2,
            batch_first=True,
        )
        self.aggr_and = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_mlp = PlainMLP(2 * dim_hidden, dim_hidden, dim_hidden, num_layer=3, p_drop=0.5)
        self.readout_prob = MLP(dim_hidden, 32, 1, num_layer=3, p_drop=0.2)
        self.pos_enc = nn.Embedding(10, dim_hidden)

    def forward(self, graph):
        device = next(self.parameters()).device
        h = self.hf_init(graph.x.float().to(device))
        h[graph.forward_level == 0] = graph.prob[graph.forward_level == 0].unsqueeze(-1).repeat(1, self.dim_hidden)

        max_level = torch.max(graph.forward_level).item() + 1
        for _ in range(self.num_rounds):
            for level in range(1, max_level):
                h_new = h.clone()
                level_mask = graph.forward_level[graph.edge_index[1]] == level
                edge_index = graph.edge_index[:, level_mask]
                if edge_index.numel() == 0:
                    h = h_new
                    continue

                target_nodes = edge_index[1]
                source_nodes = edge_index[0]
                unique_targets, inverse = torch.unique(target_nodes, return_inverse=True)
                max_length = torch.bincount(inverse).max().item()
                index_matrix = torch.full((len(unique_targets), max_length), -1, dtype=torch.long, device=device)
                padding_mask = torch.ones((len(unique_targets), max_length), dtype=torch.bool, device=device)
                source_counts = torch.bincount(inverse)
                source_offsets = torch.cumsum(source_counts, dim=0) - source_counts
                source_positions = torch.arange(source_nodes.size(0), device=device) - source_offsets[inverse]
                index_matrix[inverse, source_positions] = source_nodes
                padding_mask[inverse, source_positions] = False

                sequence = torch.cat([h[unique_targets].unsqueeze(1), h[index_matrix]], dim=1)
                padding_mask = torch.cat(
                    [torch.zeros((len(unique_targets), 1), dtype=torch.bool, device=device), padding_mask],
                    dim=1,
                )
                pos = torch.arange(sequence.shape[1], device=device).unsqueeze(0).repeat(sequence.shape[0], 1)
                updated = self.aggr_and(sequence + self.pos_enc(pos), src_key_padding_mask=padding_mask)[:, 0]
                h_new[unique_targets] = updated
                h = h_new
        return h

    def pred_prob(self, hidden):
        prob = self.readout_prob(hidden)
        return torch.clamp(prob, min=-1.0, max=1.0)


class TRACETrainer(pl.LightningModule):
    """Lightning trainer for PM netlist predictive TRACE."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = TRACE(
            num_rounds=args.num_rounds,
            dim_hidden=args.hidden,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        self.loss = nn.L1Loss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _cop(self, edge_index, prob, truth_table):
        device = edge_index.device
        target_nodes = edge_index[1]
        source_nodes = edge_index[0]
        unique_targets, inverse = torch.unique(target_nodes, return_inverse=True)
        max_length = 6
        index_matrix = torch.full((len(unique_targets), max_length), -1, dtype=torch.long, device=device)

        source_counts = torch.bincount(inverse)
        source_offsets = torch.cumsum(source_counts, dim=0) - source_counts
        source_positions = torch.arange(source_nodes.size(0), device=device) - source_offsets[inverse]
        source_positions = source_positions + max_length - source_counts[inverse]
        index_matrix[inverse, source_positions] = source_nodes

        num_patterns = 2 ** max_length
        patterns = torch.tensor([list(map(int, f'{i:0{max_length}b}')) for i in range(num_patterns)], device=device)
        aug_patterns = torch.cat([patterns, 1 - patterns], dim=1).float().unsqueeze(0).repeat(unique_targets.shape[0], 1, 1)
        input_prob = torch.cat([prob[index_matrix], 1 - prob[index_matrix]], dim=1)
        masked_input = aug_patterns * input_prob.unsqueeze(1).repeat(1, num_patterns, 1)
        masked_input = torch.where(aug_patterns == 0, 1, masked_input)
        cop_result = (torch.prod(masked_input, dim=-1) * truth_table[unique_targets].flip(-1)).sum(dim=-1)
        cov = prob[unique_targets] - cop_result
        return unique_targets, cov, cop_result

    def _cop_with_cov(self, graph, cov):
        pred_prob = torch.zeros_like(graph.prob, device=graph.prob.device)
        pred_prob[graph.forward_level == 0] = graph.prob[graph.forward_level == 0]
        max_level = torch.max(graph.forward_level)
        for level in range(1, max_level):
            level_nodes = graph.forward_index[graph.forward_level == level]
            if level_nodes.size(0) == 0:
                continue
            level_mask = graph.forward_level[graph.edge_index[1]] == level
            edge_index = graph.edge_index[:, level_mask]
            unique_targets, _, cop_result = self._cop(edge_index, pred_prob, graph.x)
            pred_prob[unique_targets] = cop_result + cov[unique_targets].squeeze(1)
        return pred_prob

    def forward(self, graph):
        hidden = self.encoder(graph)
        return self.encoder.pred_prob(hidden)

    def _shared_step(self, graph, phase):
        pred_cov = self(graph)
        unique_targets, cov, _ = self._cop(graph.edge_index, graph.prob, graph.x)
        unique_targets = graph.forward_index[graph.forward_level > 0]

        if phase == 'train':
            target = cov.unsqueeze(1)
            prediction = pred_cov[unique_targets]
        else:
            pred_prob = self._cop_with_cov(graph, pred_cov)
            target = graph.prob[unique_targets]
            prediction = pred_prob[unique_targets]

        loss = self.loss(prediction, target)
        r2_value = r2_score(target.detach().cpu().numpy(), prediction.detach().cpu().numpy())
        r2 = torch.tensor(r2_value, device=loss.device, dtype=loss.dtype)
        return loss, loss.detach(), r2

    def training_step(self, graph, batch_idx):
        loss, mae, _ = self._shared_step(graph, 'train')
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        self.log('train_mae', mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        return loss

    def validation_step(self, graph, batch_idx):
        loss, mae, r2 = self._shared_step(graph, 'val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        self.log('val_mae', mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        self.log('val_r2', r2, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        return loss

    def test_step(self, graph, batch_idx):
        loss, mae, r2 = self._shared_step(graph, 'val')
        self.log('test_mae', mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        self.log('test_r2', r2, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.args.batch_size)
        return loss
