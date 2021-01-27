import torch
import torch.nn as nn

from nlp.models.language_modeling.language_model_base import LanguageModelBase
from nlp.modules.embedding import Embedding
from nlp.constants import PAD

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
                emissions.append(logits)
        else:
            inputs = x[:,0]
            for i in range(max_seq_len):
                x_emb = self.embedding(inputs)
                hidden_state, cell_state = self.encoder(x_emb, (hidden_state, cell_state))
                logits = self.fc(hidden_state)
                emissions.append(logits)
                inputs = torch.argmax(logits, dim=1).detach()
                
        emissions = torch.cat(emissions, dim=0)
        return emissions
    
    def save_embedding(self):
        if self.hparams.model.embedding.save_numpy_filepath:
            self.embedding.save_numpy(self.hparams.model.embedding.save_numpy_filepath)
        if self.hparams.model.embedding.save_torch_filepath:
            self.embedding.save_numpy(self.hparams.model.embedding.save_torch_filepath)
            
    def save_encoder(self):
        if self.hparams.model.encoder.save_filepath:
            new_state_dict = {f"{k}_l0":v for k,v in self.encoder.state_dict().items()}
            torch.save(new_state_dict, self.hparams.model.encoder.save_filepath)
        
                
                