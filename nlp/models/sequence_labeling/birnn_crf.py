import torch
import torch.nn as nn
from nlp.modules.seq2seq_encoders import RNNSeq2SeqEncoder
from nlp.models.sequence_labeling import SequenceLabelingBase
from nlp.constants import PAD
from torchcrf import CRF


class BiRecurrentCRF(SequenceLabelingBase):
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

        self.rnn = RNNSeq2SeqEncoder(
            self.hparams.model.word_embedding_dim,
            self.hparams.model.hidden_size,
            self.hparams.model.rnn_type,
            self.hparams.model.num_layers,
            self.hparams.model.dropout,
        )
        self.fc = nn.Linear(
            2 * self.hparams.model.hidden_size, self.hparams.model.num_labels
        )
        self.crf = CRF(self.hparams.model.num_labels, batch_first=True)

        self.example_input_array = (
            torch.zeros(32, 40, dtype=torch.long),
            torch.ones(32, dtype=torch.long),
        )

    def forward(self, x, x_lens):
        outputs = self.rnn(self.word_embedding(x), x_lens)
        emissions = self.fc(outputs)
        return emissions

    def loss(self, x, x_lens, y):
        y_hat = self(x, x_lens)
        # TODO: add mask for crf based on lengths
        loss = -self.crf(y_hat, y)
        return loss

    def decode(self, x, x_lens):
        y_hat = self(x, x_lens)
        y_pred = self.crf.decode(y_hat)
        y_pred = torch.tensor(y_pred, device=y_hat.device)
        return y_pred
