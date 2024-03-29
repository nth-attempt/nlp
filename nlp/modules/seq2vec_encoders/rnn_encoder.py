import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNSeq2VecEncoder(nn.Module):
    """Some Information about RNNSeq2VecEncoder"""

    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0,
        bidirectional=True,
    ):
        super(RNNSeq2VecEncoder, self).__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, inputs, input_lengths=None):
        """
        https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        
        inputs: (batch_size, seq_len, input_size)
        input_lens: (batch_size)
        """
        if not input_lengths:
            outputs, _ = self.rnn(inputs)
        else:
            total_length = inputs.size(1)
            packed_inputs = pack_padded_sequence(
                inputs, input_lengths, batch_first=True
            )
            packed_outputs, _ = self.rnn(packed_inputs)
            outputs, _ = pad_packed_sequence(
                packed_outputs, batch_first=True, total_length=total_length,
            )
        # outputs: (batch_size, seq_len, hidden_size)
        return outputs[:, -1, :]

