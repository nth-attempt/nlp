import torch
import torch.nn as nn


class CNNSeq2VecEncoder(nn.Module):
    """Some Information about CNNSeq2VecEncoder"""

    def __init__(self):
        super(CNNSeq2VecEncoder, self).__init__()

    def forward(self, x):

        return x


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
