import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ner.modules.encoders import RNNEncoder
from ner.utils.constants import PAD
from ner.metrics import Metric

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BiLSTMCRF(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs):
        super().__init__()
        self.hparams = conf
        self.rnn_encoder = RNNEncoder(
            self.hparams.model.input_dim,
            self.hparams.model.embedding_dim,
            self.hparams.model.hidden_dim,
        )
        self.fc = nn.Linear(
            2 * self.hparams.model.hidden_dim, self.hparams.model.output_dim
        )
        self.crf = CRF(self.hparams.model.output_dim, batch_first=True)
        self.example_input_array = (
            torch.zeros(32, 40, dtype=torch.long),
            torch.ones(32, dtype=torch.long),
        )

    def forward(self, x, x_lens):
        outputs, _ = self.rnn_encoder(x, x_lens)
        emissions = self.fc(outputs)
        return emissions

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train.adam.lr,
            weight_decay=self.hparams.train.adam.weight_decay,
        )

    def step(self, batch):
        x, x_lens, y, _ = batch
        y_hat = self(x, x_lens)
        # TODO: add mask for crf based on lengths
        # loss = F.cross_entropy(
        #     y_hat.view(-1, y_hat.shape[-1]), y.view(-1), ignore_index=PAD
        # )
        loss = -self.crf(y_hat, y)
        y_pred = self.crf.decode(y_hat)
        y_pred = torch.tensor(y_pred, device=y_hat.device)
        return loss, y.detach(), y_pred.detach()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        logs = {"train_step_loss": loss}
        return {"loss": loss, "log": logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"train_loss": avg_loss}
        return {"train_loss": avg_loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss, y, y_pred = self.step(batch)
        return {"val_loss": loss, "y": y, "y_pred": y_pred}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = []
        y_pred = []
        for output in outputs:
            for instance in output["y"].tolist():
                y.append([(tag, i) for i, tag in enumerate(instance)])
            for instance in output["y_pred"].tolist():
                y_pred.append([(tag, i) for i, tag in enumerate(instance)])

        metric = Metric("val_metric", y, y_pred)
        micro_f1 = metric.micro_avg_f_score()

        logs = {"val_loss": avg_loss, "val_micro_f1": micro_f1}
        return {"val_loss": avg_loss, "log": logs, "progress_bar": logs}
