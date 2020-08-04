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

import sys
from ner.utils.gpu_profile import gpu_profile


class BiLSTMCRF(pl.LightningModule):
    def __init__(self, input_dim=23626, embedding_dim=300, hidden_dim=128, output_dim=9):
        super().__init__()
        self.output_dim = output_dim
        self.rnn_encoder = RNNEncoder(input_dim, embedding_dim, hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.crf = CRF(output_dim, batch_first=True)
        self.example_input_array = (
            torch.zeros(32, 40, dtype=torch.long),
            torch.ones(32, dtype=torch.long),
        )

    def forward(self, x, x_lens):
        outputs, _ = self.rnn_encoder(x, x_lens)
        emissions = self.fc(outputs)
        return emissions

    def training_step(self, batch, batch_idx):
        x, x_lens, y, _ = batch
        y_hat = self(x, x_lens)
        # TODO: add mask for crf based on lengths
        # loss = F.cross_entropy(
        #     y_hat.view(-1, y_hat.shape[-1]), y.view(-1), ignore_index=PAD
        # )
        loss = -self.crf(y_hat, y)
        y_pred = self.crf.decode(y_hat)
        y_pred = torch.tensor(y_pred, device=y_hat.device)

        logs = {"train_step_loss": loss}
        return {"loss": loss, "log": logs, "y": y, "y_pred": y_pred}

    def training_epoch_end(self, outputs):
        gpu_profile(frame=sys._getframe(), event="line", arg=None)
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # TODO: replace with pl metric class once automatic aggregation is ready
        y = []
        y_pred = []
        for output in outputs:
            for instance in output["y"].tolist():
                y.append([(tag, i) for i, tag in enumerate(instance)])
            for instance in output["y_pred"].tolist():
                y_pred.append([(tag, i) for i, tag in enumerate(instance)])

        metric = Metric("train_metric", y, y_pred)
        micro_f1 = metric.micro_avg_f_score()

        logs = {"train_loss": avg_loss, "train_micro_f1": micro_f1}
        return {"train_loss": avg_loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, x_lens, y, _ = batch
        y_hat = self(x, x_lens)
        # loss = F.cross_entropy(
        #     y_hat.view(-1, y_hat.shape[-1]), y.view(-1), ignore_index=PAD
        # )
        loss = -self.crf(y_hat, y)
        y_pred = self.crf.decode(y_hat)
        y_pred = torch.tensor(y_pred, device=y_hat.device)

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)
