import torch
import torch.nn as nn
from nlp.constants import PAD
from nlp.models.sequence_labeling import SequenceLabelingBase
from nlp.modules.seq2seq_encoders import RNNSeq2SeqEncoder
from nlp.modules.seq2vec_encoders import CNNSeq2VecEncoder, RNNSeq2VecEncoder
from torchcrf import CRF


class BiRNN(SequenceLabelingBase):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)
        self.word_embedding = nn.Embedding(
            self.hparams.model.num_words,
            self.hparams.model.word_embedding_dim,
            padding_idx=PAD,
        )
        # if isinstance(pretrained_word_embedding, torch.Tensor):
        #     self.word_embedding.weight.data.copy_(pretrained_word_embedding)
        #     self.word_embedding.weight.requires_grad = requires_grad
        if self.hparams.model.use_char:
            self.char_embedding = nn.Embedding(
                self.hparams.model.num_chars,
                self.hparams.model.char_embedding_dim,
                padding_idx=PAD,
            )
            if self.hparams.model.char_encoder_type == "RNN":
                self.char_encoder = RNNSeq2VecEncoder(
                    self.hparams.model.char_embedding_dim,
                    self.hparams.model.char_rnn.hidden_size,
                    self.hparams.model.char_rnn.rnn_type,
                    self.hparams.model.char_rnn.num_layers,
                    self.hparams.model.char_rnn.dropout,
                )
            else:
                self.char_encoder = CNNSeq2VecEncoder()
        # TODO: RNN input dropout and RNN output dropout
        self.rnn = RNNSeq2SeqEncoder(
            self.hparams.model.word_embedding_dim
            + self.hparams.model.char_rnn.hidden_size * 2,
            self.hparams.model.word_rnn.hidden_size,
            self.hparams.model.word_rnn.rnn_type,
            self.hparams.model.word_rnn.num_layers,
            self.hparams.model.word_rnn.dropout,
        )
        self.fc = nn.Linear(
            2 * self.hparams.model.word_rnn.hidden_size,
            self.hparams.model.num_labels,
        )
        self.crf = CRF(self.hparams.model.num_labels, batch_first=True)

        # self.example_input_array = (
        #     torch.zeros(32, 40, dtype=torch.long),
        #     torch.ones(32, dtype=torch.long),
        # )

    def forward(self, words, seq_lens, chars):
        words_embedded = self.word_embedding(words)
        if self.hparams.model.use_char:
            bs, l, lc = chars.size()
            chars_flat = chars.view(-1, lc)
            chars_embedded = self.char_embedding(chars_flat)
            char_feats = self.char_encoder(
                chars_embedded
            )  # what about char lengths
            char_feats = char_feats.view(bs, l, -1)
            word_feats = torch.cat((words_embedded, char_feats), dim=2)
        else:
            word_feats = words_embedded
        outputs = self.rnn(word_feats, seq_lens)
        emissions = self.fc(outputs)
        return emissions
        # outputs = self.rnn(self.word_embedding(x), x_lens)
        # emissions = self.fc(outputs)
        # return emissions

    def loss(self, x, x_lens, y, chars):
        mask = torch.ne(x, PAD)
        y_hat = self(x, x_lens, chars)
        loss = -self.crf(y_hat, y, mask=mask)
        return loss

    def decode(self, x, x_lens, chars):
        mask = torch.ne(x, PAD)
        y_hat = self(x, x_lens, chars)
        y_pred = self.crf.decode(y_hat, mask)
        return y_pred
