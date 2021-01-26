import torch
import torch.nn as nn

import pytorch_lightning as pl

from nlp.metrics import Metric
from nlp.constants import PAD

class LanguageModelBase(pl.LightningModule):
    
    def __init__(
        self,
        hparams,
        *args,
        **kwargs,
    ):
        
        super().__init__()
        self.save_hyperparameters(hpramas)
        self.teacher_forcing_rate = self.hparams.model.teacher_forcing_rate
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)