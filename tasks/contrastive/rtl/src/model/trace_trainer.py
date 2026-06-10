import torch
import pytorch_lightning as pl
from info_nce import InfoNCE

from .trace_encoder import TRACE


class TRACETrainer(pl.LightningModule):
    """Lightning trainer for RTL contrastive TRACE."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.encoder_type = 'TRACE'
        self.args.aggr = 'transformer'
        self.encoder = TRACE(args, dim_hidden=args.hidden)
        self.infonce = InfoNCE(negative_mode='paired')
        self.max_list_len = 512
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.val_step_outputs = []
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def _recall_at_k(self, scores, k):
        target = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        _, topk = torch.topk(scores, k, dim=1)
        return (topk == target.unsqueeze(1)).any(dim=1).float().mean().item()

    def _graph_embedding(self, graph):
        node_h = self.encoder(graph, input_pattern=None)
        forward_level = graph.forward_level.to(node_h.device)
        output_mask = forward_level == forward_level.max()
        if output_mask.any():
            return node_h[output_mask].mean(dim=0, keepdim=True)
        return node_h.mean(dim=0, keepdim=True)

    def forward_cls(self, batch):
        bs = batch.batch_size
        device = next(self.parameters()).device
        orig_h = torch.cat([self._graph_embedding(batch.gate[i]) for i in range(bs)], dim=0)
        syn_h = torch.cat([self._graph_embedding(batch.syn_gate[i]) for i in range(bs)], dim=0)

        neg_idx = torch.arange(bs, device=device).unsqueeze(0).repeat(bs, 1)
        neg_idx = neg_idx[~torch.eye(bs, dtype=torch.bool, device=device)].view(bs, bs - 1)
        if bs > self.max_list_len:
            neg_idx = neg_idx[:, torch.randperm(bs - 1, device=device)[:self.max_list_len]]
        loss = self.infonce(orig_h, syn_h, syn_h[neg_idx])

        pos_sim = torch.cosine_similarity(orig_h, syn_h, dim=-1).unsqueeze(1)
        neg_sim = torch.cosine_similarity(orig_h.unsqueeze(1).expand(-1, neg_idx.shape[1], -1), syn_h[neg_idx], dim=-1)
        scores = torch.cat([pos_sim, neg_sim], dim=1)
        metrics = {'R@1': self._recall_at_k(scores, 1), 'R@5': self._recall_at_k(scores, min(5, scores.size(1))), 'R@10': self._recall_at_k(scores, min(10, scores.size(1)))}
        return loss, metrics

    def training_step(self, batch, batch_idx):
        loss, metrics = self.forward_cls(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.training_step_outputs.append({'loss': loss.detach(), 'metrics': metrics})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.forward_cls(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.val_step_outputs.append({'loss': loss.detach(), 'metrics': metrics})
        return loss

    def test_step(self, batch, batch_idx):
        loss, metrics = self.forward_cls(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.test_step_outputs.append({'loss': loss.detach(), 'metrics': metrics})
        return loss

    def _log_epoch(self, outputs, prefix):
        if not outputs:
            return
        loss = sum(item['loss'] for item in outputs) / len(outputs)
        r1 = sum(item['metrics']['R@1'] for item in outputs) / len(outputs)
        r5 = sum(item['metrics']['R@5'] for item in outputs) / len(outputs)
        r10 = sum(item['metrics']['R@10'] for item in outputs) / len(outputs)
        self.log(f'{prefix}_loss_epoch', round(float(loss), 4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log(f'{prefix}_R@1_epoch', round(float(r1), 4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log(f'{prefix}_R@5_epoch', round(float(r5), 4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        self.log(f'{prefix}_R@10_epoch', round(float(r10), 4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        outputs.clear()

    def on_train_epoch_end(self):
        self._log_epoch(self.training_step_outputs, 'train')

    def on_test_epoch_end(self):
        self._log_epoch(self.test_step_outputs, 'test')

    def on_validation_epoch_end(self):
        self._log_epoch(self.val_step_outputs, 'val')
