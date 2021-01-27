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
        
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train.adam.lr,
            weight_decay=self.hparams.train.adam.weight_decay,
        )
    
    def __score(self, outputs):
        y_true = []
        y_pred = []
        for output in outputs:
            y_true.append([(tag,i) for i,tag in enumerate(output["y"].tolist())])
            y_pred.append([(tag,i) for i,tag in enumerate(output["y_pred"].tolist())])
        metric = Metric("micro avg")
        metric.score(y_true, y_pred, ignore_classes=[])
        micro_f1 = metric.micro_avg_f_score()
        macro_f1 = metric.macro_avg_f_score()
        
        return micro_f1, macro_f1
    
    def step(
        self,
        batch,
        toss=0.0
    ):
        x, x_lens, y = batch
        y_hat = self(x, x_lens, toss)
        
        # flatten y_hat and y for loss and metrcs
        y = torch.squeeze(y.view(-1, 1))
        y_hat = y_hat.view(-1, self.hparams.data.output_vocab_size)
        # loss and preds
        loss = self.loss_fn(y_hat, y)
        y_pred = torch.argmax(y_hat, dim=1).detach()
        
        return loss, y.detach(), y_pred.detach()
    
    def training_step(
        self,
        batch,
        batch_idx
    ):
        toss = torch.rand(1)
        loss, y, y_pred = self.step(batch, toss)
        perplexity = torch.exp(loss)
        self.log(
            "train_loss", 
            loss, 
            on_epoch=True, 
            logger=True,
        )
        return {
            "loss": loss,
            "perplexity": perplexity,
        }
    
    def validation_step(
        self,
        batch,
        batch_idx,
    ):
        toss = 1.0 # validation always decoding
        loss, y, y_pred = self.step(batch, toss)
        perplexity = torch.exp(loss)
        self.log(
            "train_loss", 
            loss, 
            prog_bar=True, 
            logger=True,
        )
        return {
            "y": y,
            "y_pred": y_pred,
            "val_loss": loss,
            "val_perplexity": perplexity,
        }
    
    def training_epoch_end(
        self,
        outputs,
    ):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_perplexity = torch.stack([x["perplexity"] for x in outputs]).mean()
        
    def validation_epoch_end(
        self,
        outputs,
    ):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_perplexity = torch.stack([x["val_perplexity"] for x in outputs]).mean()
        micro_f1, macro_f1 = self.__score(outputs)
        self.log("val_micro_f1", micro_f1, prog_bar=True)
        self.log("val_loss", avg_loss, prog_bar=True)
        
    def test_step(
        self,
        batch,
        batch_idx,
    ):
        toss = 1
        loss, y_true, y_pred = self.step(batch, toss)
        perplexity = torch.exp(loss)
        return {
            "test_loss": loss,
            "y": y_true,
            "y_pred": y_pred,
            "test_perplexity": perplexity,
        }
    
    def test_epoch_end(
        self,
        outputs,
    ):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_perplexity = torch.stack([x["val_perplexity"] for x in outputs]).mean()
        micro_f1, macro_f1 = self.__score(outputs)
        logs = {
            "test_loss": avg_loss,
            "test_perplexity": avg_perplexity,
            "test_micro_f1": micro_f1,
            "test_macro_f1": macro_f1,
        }
        return {
            "test_loss": avg_loss,
            "test_perplexity": avg_perplexity,
            "log": logs,
            "progress_bar": logs,
        }
        
        