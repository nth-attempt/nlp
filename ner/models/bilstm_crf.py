import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ner.modules.encoders import RNNEncoder
from ner.utils.constants import PAD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMCRF(pl.LightningModule):
    def __init__(self, input_dim=23626, embedding_dim=300, hidden_dim=128, output_dim=9):
        super().__init__()
        self.rnn_encoder = RNNEncoder(input_dim, embedding_dim, hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.example_input_array = torch.zeros(32, 40, dtype=torch.long)

    def forward(self, x, x_lens):
        outputs, _ = self.rnn_encoder(x, x_lens)
        logits = self.fc(outputs)
        return logits

    def training_step(self, batch, batch_idx):
        x, x_lens, y, _ = batch
        y_hat = self(x, x_lens)
        loss = F.cross_entropy(
            y_hat.view(-1, y_hat.shape[-1]), y.view(-1), ignore_index=PAD
        )
        logs = {"train_step_loss": loss}
        return {"loss": loss, "log": logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"train_loss": avg_loss}
        return {"train_loss": avg_loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, x_lens, y, _ = batch
        y_hat = self(x, x_lens)
        loss = F.cross_entropy(
            y_hat.view(-1, y_hat.shape[-1]), y.view(-1), ignore_index=PAD
        )
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0005)
