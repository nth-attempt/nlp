import pytorch_lightning as pl
import torch
from nlp.metrics import Metric
from nlp.constants import PAD_TOKEN, O_TOKEN


class SequenceLabelingBase(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.label_vocab = kwargs.get("label_vocab")

    def training_step(self, batch, batch_idx):
        x, x_lens, y, chars = batch
        loss = self.loss(x, x_lens, y, chars)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_pred = self._prediction_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return {"y": y, "y_pred": y_pred}

    def validation_epoch_end(self, validation_step_outputs):
        y, y_pred = self._prediction_epoch_end(validation_step_outputs)
        micro_f1 = self._score(y, y_pred)
        self.log("val_micro_f1", micro_f1, prog_bar=True)
        # save the hparams to tune
        self.logger.log_hyperparams(self.hparams, {"micro_f1": micro_f1})

    def test_step(self, batch, batch_idx):
        loss, y, y_pred = self._prediction_step(batch, batch_idx)
        self.log("test_loss", loss)
        return {"y": y, "y_pred": y_pred}

    def test_epoch_end(self, test_step_outputs):
        y, y_pred = self._prediction_epoch_end(test_step_outputs)
        micro_f1 = self._score(y, y_pred)
        self.log("test_micro_f1", micro_f1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.train.adam.lr,
            weight_decay=self.hparams.train.adam.weight_decay,
        )

    def _prediction_step(self, batch, batch_idx):
        x, x_lens, y, chars = batch
        loss = self.loss(x, x_lens, y, chars)
        y_pred = self.decode(x, x_lens, chars)
        # y_pred is a List[List]
        y_pred = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(i, device=self.device) for i in y_pred],
            batch_first=True,
        )
        return loss, y, y_pred

    def _prediction_epoch_end(self, prediction_step_outputs):
        y = []
        y_pred = []
        for output in prediction_step_outputs:
            for instance in output["y"].tolist():
                y.append([(tag, i) for i, tag in enumerate(instance)])
            for instance in output["y_pred"].tolist():
                y_pred.append([(tag, i) for i, tag in enumerate(instance)])
        return y, y_pred

    def _score(self, y, y_pred):
        metric = Metric("micro avg")
        metric.score(
            y,
            y_pred,
            ignore_classes=[
                self.label_vocab.get_id(PAD_TOKEN),
                self.label_vocab.get_id(O_TOKEN),
            ],
        )
        return metric.micro_avg_f_score()

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        pass
