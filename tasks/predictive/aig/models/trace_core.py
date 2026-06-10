import torch
from torch import nn
import pytorch_lightning as pl

from .cov2prob import cov2prob
from .mlp import MLP
from .trace_encoder import TRACE


class TRACECore(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = TRACE(args, num_rounds=args.num_layers, dim_hidden=args.hidden)
        self.readout_prob = self.get_decoder(args.hidden, 1)
        self.l1_loss = nn.L1Loss()
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.val_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def get_decoder(self, dim_in, dim_out):
        return MLP(
            dim_in=dim_in,
            dim_hidden=dim_in // 2,
            dim_pred=dim_out,
            num_layer=3,
            norm_layer='batchnorm',
            act_layer='silu',
            p_drop=0.2,
        )

    def forward(self, batch, inference=False):
        hidden = self.encoder(batch)
        prediction = self.readout_prob(hidden).squeeze(-1)

        if self.args.learning_target == 'probability':
            pred_prob = torch.clamp(prediction, 0.0, 1.0)
            return self.l1_loss(batch.prob, pred_prob)

        if self.args.learning_target == 'covariance':
            pred_cov = torch.clamp(prediction, -1.0, 1.0)
            if inference:
                pred_prob = cov2prob(batch, pred_cov)
                return self.l1_loss(batch.prob, pred_prob)
            and_mask = batch.gate.squeeze() == 1
            return self.l1_loss(batch.cov[and_mask], pred_cov[and_mask])

        raise NotImplementedError('Learning target not implemented')

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, inference=False)
        self.log('train_prob_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.training_step_outputs.append({'prob_loss': loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch, inference=True)
        self.log('val_prob_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.val_step_outputs.append({'prob_loss': loss.item()})
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.forward(batch, inference=True)
        self.log('test_prob_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=self.args.batch_size)
        self.test_step_outputs.append({'prob_loss': loss.item()})
        return loss

    def _log_epoch(self, outputs, prefix):
        if not outputs:
            return
        prob_loss = sum(item['prob_loss'] for item in outputs) / len(outputs)
        self.log(f'{prefix}_prob_loss_epoch', round(float(prob_loss), 4), on_epoch=True, prog_bar=True, logger=True, batch_size=self.args.batch_size)
        outputs.clear()

    def on_train_epoch_end(self):
        self._log_epoch(self.training_step_outputs, 'train')

    def on_test_epoch_end(self):
        self._log_epoch(self.test_step_outputs, 'test')

    def on_validation_epoch_end(self):
        self._log_epoch(self.val_step_outputs, 'val')
