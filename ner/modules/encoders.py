import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ner.utils.constants import PAD


class RNNEncoder(nn.Module):
    """Some Information about RNNEncoder"""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        rnn_type: str = "LSTM",
        bidirectional: bool = True,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0,
    ):
        super(RNNEncoder, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=PAD)
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=self.batch_first,
        )

    def forward(self, x, x_lens):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(
            embedded, x_lens, batch_first=self.batch_first
        )
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=self.batch_first)
        return outputs, hidden
