import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp.modules.encoders import RNNEncoder
from nlp.utils.constants import PAD
from nlp.metrics import Metric

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
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

    def forward(self, x, x_lens):
        outputs, _ = self.rnn_encoder(x, x_lens)
        emissions = self.fc(outputs)
        return emissions

    def loss(self, x, x_lens, y):
        y_hat = self(x, x_lens)
        # TODO: add mask for crf based on lengths
        loss = -self.crf(y_hat, y)
        return loss

    def decode(self, x, x_lens):
        y_hat = self(x, x_lens)
        y_pred = self.crf.decode(y_hat)
        y_pred = torch.tensor(y_pred, device=y_hat.device)
        return y_pred
