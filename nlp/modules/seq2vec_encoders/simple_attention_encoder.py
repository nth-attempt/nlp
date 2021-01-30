import torch
import torch.nn as nn

class SimpleAttentionEncoder(nn.Module):
    
    def __init__(
        self,
        input_size,
        attention_size,
    ):
        super(SimpleAttentionEncoder, self).__init__()
        
        self.att_fc = nn.Linear(input_size, attention_size)
        self.att_vec = torch.rand(attention_size, requires_grad=True)
        
    def forward(self, inputs, mask):
        """
        inputs = batch_size x max_seq_len x input_size
        masks = batch_size x max_seq_len
        """
        batch_size = inputs.shape[0]
        max_seq_len = inputs.shape[1]
        
        wx_plus_b = self.att_fc(inputs) # wx + b = batch_size x max_seq_len x attention_size
        wx_plus_b = wx_plus_b.view((batch_size*max_seq_len, attention_size)) # batch_size*max_seq_len x attention_size
        
        # each wx_plus_b.att_vec gives one scalar. so batch_size*max_seq_len scalars
        att_scores = torch.sum(torch.mul(wx_plus_b, self.att_vec), dim=1, keepdim=False) # batch_size*max_seq_len x 1
        att_scores = att_scores.view((batch_size, max_seq_len)) # batch_size x max_seq_len
        # https://github.com/allenai/allennlp/blob/main/allennlp/nn/util.py#L270
        att_weights = nn.functional.softmax(att_scores*mask, dim=1)
        att_weights = att_weights * att_weights
        att_weights = att_weights / att_weights.sum(dim=1, keepdim=True) # batch_size x max_seq_len
        
        # multiply attention weights and inputs, then sum them over time dimension
        # get one vector per data point of size input_size
        encoded_inputs = torch.sum(torch.mul(inputs, att_weights.unsqueeze(-1)), dim=1) # batch_size x input_size
        
        return encoded_inputs