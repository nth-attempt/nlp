import os
import torch
import torch.nn as nn
import numpy as np
from nlp.constants import PAD

class Embedding(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        freeze_weights: bool = False,
        filepath: str = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=PAD)
        self.__init_weights(freeze_weights, filepath)
        
    def __init_weights(
        self, 
        freeze_weights=False, 
        filepath=None
    ):
        
        if not filepath:
            initrange = 1.0
            nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        else:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"{filepath} not found")
            try:
                self.load_numpy(filepath)
            except:
                try:
                    self.load_torch(filepath)
                except IOError:
                    raise IOError(f"could not read {filepath} using numpy.load and load_state_dict(torch.load)")
        self.embedding.weight.requires_grad = not freeze_weights
        
    def forward(
        self,
        x
    ):
        embedded = self.embedding(x)
        return embedded
    
    def save_torch(
        self,
        filepath,
    ):
        torch.save(self.embedding.state_dict(), filepath)
        
    def load_torch(
        self,
        filepath,
    ):
        self.embedding.load_state_dict(torch.load(filepath))
        
    def save_numpy(
        self,
        filepath,
    ):
        np_embedding = self.embedding.weight.detach().cpu().numpy()
        np.save(filepath, np_embedding)
        
    def load_numpy(
        self,
        filepath,
    ):
        pretrained_embedding = np.load(filepath)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embedding))