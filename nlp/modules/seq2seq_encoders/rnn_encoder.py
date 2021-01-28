import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNSeq2SeqEncoder(nn.Module):
    """Outputs the hidden state for every time step

    `batch_first` is always True.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        rnn_type="LSTM",
        num_layers=1,
        dropout=0.0,
        bidirectional=True,
        freeze_weights=False,
        filepath=None,
        batch_first=True,
    ):
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.length_dim = int(self.batch_first)
        
        super(RNNSeq2SeqEncoder, self).__init__()
        
        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=self.batch_first,
            dropout=dropout,
            bidirectional=self.bidirectional,
        )
        
        self.__init_weights(freeze_weights, filepath)
        
    def __init_weights(
        self,
        freeze_weights=False,
        filepath=None,
    ):
        if filepath:
            self.load_torch(filepath)
            
        for param in self.rnn.parameters():
            param.requires_grad = not freeze_weights
            
    def forward(
        self,
        inputs,
        input_lengths,
        h0=None, # allow for passing some learnable h0/c0
        c0=None,
    ):
        total_length = inputs.size(self.length_dim)
        batch_size = inputs.shape[int(not self.batch_first)]
        if h0==None:
            h0 = torch.zeros(((1+int(self.bidirectional))*self.num_layers, batch_size, self.hidden_size), device=inputs.device)
            c0 = torch.zeros(((1+int(self.bidirectional))*self.num_layers, batch_size, self.hidden_size), device=inputs.device)
            
                       
        packed_inputs = pack_padded_sequence(inputs, input_lengths, batch_first=self.batch_first)
        packed_outputs, hiddens = self.rnn(packed_inputs, (h0, c0))
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=self.batch_first, total_length=total_length)
        
        return outputs, hiddens
    
    def save_torch(self, filepath):
        torch.save(self.rnn.state_dict(), filepath)
        
    def load_torch(self, filepath):
        self.rnn.load_state_dict(torch.load(filepath))
        