import torch
import torch.nn as nn

from nlp.models.autoencoder.autoencoder_base import AutoencoderBase
from nlp.modules.seq2seq_encoders import RNNSeq2SeqEncoder
from nlp.modules.embedding import Embedding
from nlp.constants import PAD

class RNNAutoencoder(AutoencoderBase):
    
    def __init__(
        self,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__(hparams, *args, **kwargs)
        self.hparams = hparams
        self.batch_first = True
        self.use_bos = self.hparams.data.use_bos
        
        self.embedding = Embedding(
            self.hparams.data.input_vocab_size,
            self.hparams.model.embedding.embedding_size,
            self.hparams.model.embedding.freeze_weights,
            self.hparams.model.embedding.load_filepath,
        )
        
        self.encoder = RNNSeq2SeqEncoder(
            self.hparams.model.embedding.embedding_size,
            self.hparams.model.encoder.hidden_size,
            self.hparams.model.encoder.rnn_type,
            self.hparams.model.encoder.num_layers,
            self.hparams.model.encoder.dropout,
            self.hparams.model.encoder.bidirectional,
            self.hparams.model.encoder.freeze_weights,
            self.hparams.model.encoder.load_filepath,
            batch_first=True,
        )
        
        cell_type = f"{self.hparams.model.decoder.rnn_type}Cell"
        self.decoder = getattr(nn, cell_type)(
            self.hparams.model.embedding.embedding_size,
            self.hparams.model.decoder.hidden_size,
        )
        
        self.fc = nn.Linear(
            self.hparams.model.decoder.hidden_size,
            self.hparams.data.output_vocab_size
        )
        
    def forward(self, enc_x, dec_x, enc_lens, dec_lens, toss=1.0):
        
        batch_size = enc_x.shape[0]
        max_seq_len = enc_x.shape[1]
        if self.use_bos:
            max_seq_len += 1
        
        encoder_hidden_size = self.hparams.model.encoder.hidden_size
        encoder_num_layers = self.hparams.model.encoder.num_layers
        teacher_forcing = toss<self.teacher_forcing_rate
        
        enc_emb_x = self.embedding(enc_x)
        
        _, (encoder_hn, encoder_cn) = self.encoder(enc_emb_x, enc_lens)
        
        if self.hparams.model.encoder.bidirectional:
            encoder_hn = encoder_hn.view(encoder_num_layers, 2, batch_size, encoder_hidden_size)
            encoder_hn = torch.cat((encoder_hn[encoder_num_layers-1,:1,:,:], encoder_hn[encoder_num_layers-1,1:,:,:]), dim=2)
            encoder_cn = encoder_cn.view(encoder_num_layers, 2, batch_size, encoder_hidden_size)
            encoder_cn = torch.cat((encoder_cn[encoder_num_layers-1,:1,:,:], encoder_cn[encoder_num_layers-1,1:,:,:]), dim=2)
        else:
            encoder_hn = encoder_hn.view(encoder_num_layers, 1, batch_size, encoder_hidden_size)
            encoder_hn = encoder_hn[encoder_num_layers-1,:,:,:]
            encoder_cn = encoder_cn.view(encoder_num_layers, 1, batch_size, encoder_hidden_size)
            encoder_cn = encoder_cn[encoder_num_layers-1,:,:,:]
        
        encoder_hn = encoder_hn.squeeze(dim=0)
        encoder_cn = encoder_cn.squeeze(dim=0)
        
        emissions = []
        if teacher_forcing:
            dec_x_emb = self.embedding(dec_x)
            for i in range(max_seq_len):
                encoder_hn, encoder_cn = self.decoder(dec_x_emb[:,i,:], (encoder_hn, encoder_cn))
                logits = self.fc(encoder_hn)
                emissions.append(torch.unsqueeze(logits, 1))
        else:
            inputs = dec_x[:,0]
            for i in range(max_seq_len):
                dec_x_emb = self.embedding(inputs)
                encoder_hn, encoder_cn = self.decoder(dec_x_emb, (encoder_hn, encoder_cn))
                logits = self.fc(encoder_hn)
                emissions.append(torch.unsqueeze(logits, 1))
                inputs = torch.argmax(logits, dim=1).detach()
                
        emissions = torch.cat(emissions, dim=1)
        return emissions
    
    def save_embedding(self):
        if self.hparams.model.embedding.save_numpy_filepath:
            self.embedding.save_numpy(self.hparams.model.embedding.save_numpy_filepath)
        if self.hparams.model.embedding.save_torch_filepath:
            self.embedding.save_torch(self.hparams.model.embedding.save_torch_filepath)
            
    def save_encoder(self):
        if self.hparams.model.encoder.save_filepath:
            self.encoder.save_torch(self.hparams.model.encoder.save_filepath)
        