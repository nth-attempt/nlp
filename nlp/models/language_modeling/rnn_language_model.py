import torch
import torch.nn as nn

from nlp.models.language_modeling.language_model_base import LanguageModelBase
from nlp.modules.embedding import Embedding
from nlp.constants import PAD
from nlp.modules.seq2seq_encoders import RNNSeq2SeqEncoder

class RNNLanguageModel(LanguageModelBase):
    
    def __init__(
        self,
        hparams,
        *args,
        **kwargs,
    ):
        super().__init__(hparams, *args, **kwargs)
        self.hparams = hparams
        
        self.embedding = Embedding(
            self.hparams.data.input_vocab_size,
            self.hparams.model.embedding.embedding_size,
            self.hparams.model.embedding.freeze_weights,
            self.hparams.model.embedding.load_filepath,
        )
        
        cell_type = f"{self.hparams.model.encoder.rnn_type}Cell"
        self.encoder = getattr(nn, cell_type)(
            self.hparams.model.embedding.embedding_size,
            self.hparams.model.encoder.hidden_size,
        )
        
        self.fc = nn.Linear(
            self.hparams.model.encoder.hidden_size,
            self.hparams.data.output_vocab_size
        )
        self.hidden_size = self.hparams.model.encoder.hidden_size
        
        self.__init_weights()
        
    def __init_weights(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        
    def forward(
        self,
        x,
        x_lens,
        toss=1.0,
    ):
        batch_size = x.shape[0]
        max_seq_len = x.shape[1]
        hidden_state = torch.zeros((batch_size, self.hidden_size), device=x.device)
        cell_state = torch.zeros((batch_size, self.hidden_size), device=x.device)
        teacher_forcing = toss<self.teacher_forcing_rate
        emissions = []
        if teacher_forcing:
            x_emb = self.embedding(x)
            for i in range(max_seq_len):
                hidden_state, cell_state = self.encoder(x_emb[:,i,:], (hidden_state, cell_state))
                logits = self.fc(hidden_state)
                emissions.append(torch.unsqueeze(logits, 1))
        else:
            inputs = x[:,0]
            for i in range(max_seq_len):
                x_emb = self.embedding(inputs)
                hidden_state, cell_state = self.encoder(x_emb, (hidden_state, cell_state))
                logits = self.fc(hidden_state)
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
            new_state_dict = {f"{k}_l0":v for k,v in self.encoder.state_dict().items()}
            torch.save(new_state_dict, self.hparams.model.encoder.save_filepath)
        

'''
class RNNLanguageModel(LanguageModelBase):
    
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(hparams, *args, **kwargs)
        self.hparams = hparams
        
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
        
        self.fc = nn.Linear(
            self.hparams.model.encoder.hidden_size,
            self.hparams.data.output_vocab_size
        )
        
    def forward(
        self,
        x,
        x_lens,
        toss=1.0
    ):
        batch_size = x.shape[0]
        max_seq_len = x.shape[1]
        teacher_forcing = toss<self.teacher_forcing_rate
        
        if teacher_forcing:
            x_emb = self.embedding(x)
            encoder_outputs, encoder_states = self.encoder(x_emb, x_lens)
            #encoder_hn, encoder_cn = encoder_states
            emissions = self.fc(encoder_outputs)
        else:
            new_x = x[:,:1].detach()
            emissions = torch.zeros((batch_size, max_seq_len, self.hparams.data.output_vocab_size), device=self.device)
            for i in range(max_seq_len):
                l = i + 1
                lens = torch.tensor([l for j in range(batch_size)])
                x_emb = self.embedding(new_x)
                encoder_outputs, encoder_states = self.encoder(x_emb, lens)
                emiss = self.fc(encoder_outputs)
                y_pred = torch.argmax(emiss, dim=2).detach()
                new_x = torch.cat((new_x, y_pred[:,i:i+1]), dim=1).detach()
                emissions[:,i:i+1,:] = emiss[:,i:i+1,:]
            del new_x
            del y_pred
            del lens
        return emissions
'''         
                