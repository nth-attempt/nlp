import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nlp.utils.constants import PAD


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

        self.embedding = nn.Embedding(
            input_dim, embedding_dim, padding_idx=PAD
        )
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, x, x_lens):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(
            embedded, x_lens, batch_first=self.batch_first
        )
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=self.batch_first
        )
        return outputs, hidden


class CharCNNEncoder(nn.Module):
    """Some Information about CharCNNEncoder"""

    def __init__(self, input_dim, embedding_dim, num_filters, dropout=0.33):
        super(CharCNNEncoder, self).__init__()
        self.init_embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_conv_layers()

    def init_embedding(self, input_dim, embedding_dim):
        self.embedding = nn.Embedding(
            input_dim, embedding_dim, padding_idx=PAD
        )
        scale = math.sqrt(3.0 / embedding_dim)
        self.embedding.weight.data[1:, :].uniform_(-scale, scale)

    def init_conv_layers(
        self, embedding_dim, num_filters, min_kernel_size, max_kernel_size
    ):
        conv_layers = []
        for k in range(min_kernel_size, max_kernel_size + 1):
            layer = nn.Sequential(
                nn.Conv1d(
                    embedding_dim, num_filters, kernel_size=k, padding=1
                ),
                nn.ReLU(),
            )
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        # x -> [batch size, sent_length, char_length]
        batch_size, sent_len, char_len = x.size()
        x = x.view(-1, char_len)
        embedded = self.dropout(self.embedding(x))
        for layer in self.conv_layers:
            outputs = layer(embedded)
            outputs, _ = outputs.max(dim=2)

        # do max pooling
        return outputs.view(batch_size, sent_len, -1)


class CharRNNEncoder(nn.Module):
    """Some Information about CharRNNEncoder"""

    def __init__(self):
        super(CharRNNEncoder, self).__init__()

    def forward(self, x):

        return x
