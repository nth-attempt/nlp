import pytorch_lightning as pl
import torch
import torch.nn as nn
from nlp.modules.seq2vec_encoders import RNNSeq2VecEncoder, CNNSeq2VecEncoder
from nlp.modules.seq2seq_encoders import RNNSeq2SeqEncoder
from nlp.models.sequence_labeling import SequenceLabelingBase
from nlp.utils.constants import PAD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF


class BiRecurrentConvCRF(SequenceLabelingBase):
    def __init__(
        self,
        num_words,
        word_embedding_dim,
        use_char=None,
        num_chars=None,
        char_embedding_dim=0,
        char_encoder_type=None,
        rnn_type="LSTM",
        rnn_hidden_size=300,
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(
            num_words, word_embedding_dim, padding_idx=PAD
        )
        if isinstance(pretrained_word_embedding, torch.Tensor):
            self.word_embedding.weight.data.copy_(pretrained_word_embedding)
            self.word_embedding.weight.requires_grad = requires_grad

        self.use_char = use_char

        if self.use_char:
            self.char_embedding = nn.Embedding(
                num_chars, char_embedding_dim, padding_idx=PAD
            )
            if self.char_encoder_type == "RNN":
                self.char_encoder = RNNSeq2VecEncoder()
            else:
                self.char_encoder = CNNSeq2VecEncoder()
        # Pass char embeddings via Seq2Vec encoder
        # concat embeddings
        # run them through RNN
        self.rnn = RNNSeq2SeqEncoder(
            word_embedding_dim + char_embedding_dim,
            rnn_hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, inputs, input_lengths):
        word_feats = self.word_embedding(inputs)
        char_embedded = self.char_embedding(inputs)
        # pass char_embedded into char_encoder
        # run rnn
        return x

