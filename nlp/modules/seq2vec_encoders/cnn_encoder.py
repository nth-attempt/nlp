import torch
import torch.nn as nn


class CNNSeq2VecEncoder(nn.Module):
    """CNNSeq2VecEncoder
    !!! Not accurate as masks need to be taken into account. see https://github.com/allenai/allennlp/blob/main/allennlp/modules/seq2vec_encoders/cnn_encoder.py
    """

    def __init__(
        self,
        input_dim,
        num_filters_list,
        filter_size_list,
        activation_fn,
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros',
    ):
        super(CNNSeq2VecEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_filters_list = num_filters_list
        self.filter_size_list = filter_size_list
        self.activation_fn = activation_fn
        self.output_dim = sum(self.num_filters_list)
        
        self.conv_layers = [nn.Conv1d(self.input_dim, num_filters, filter_size, stride, padding, dilation, groups, bias, padding_mode) for num_filters,filter_size in zip(self.num_filters_list, self.filter_size_list)]

    def forward(self, x):
        """
        x : batchsize, seq_len, input_dim. 
        When using on chars instead of words, x : batchsize*seq_len, tok_len, input_dim
        """
        
        x = torch.transpose(x, 1, 2) # x : batchsize, input_dim, seq_len, so in_channels = input_dim
        output = []
        for conv_layer in self.conv_layers:
            op = conv_layer(x) # op : batchsize, num_filters, new_seq_len (calculated using padding, stride, seq_len...)
            if self.activation_fn:
                op = self.activation_fn(op)
            op = op.max(dim=2)[0] # op : batch_size, num_filters -- take the max along the dim=2 (along the seq)
            output.append(op)
        output = torch.cat(output, dim=1) # output : sum([num_filters_list])
        return output


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
