import pytorch_lightning as pl
import torch
from nlp.metrics import Metric
from nlp.utils.constants import PAD_TOKEN, O_TOKEN
from nlp.data.vocab import Vocab


class TokenClassificationTask(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, label_vocab: Vocab):
        super().__init__()
        self.model = model
        self.label_vocab = label_vocab
        self.save_hyperparameters(self.model.hparams)
        # self.example_input_array = (
        #     torch.zeros(32, 40, dtype=torch.long),
        #     torch.ones(32, dtype=torch.long),
        # )

    def training_step(self, batch, batch_idx):
        x, x_lens, y, _ = batch
        loss = self.model.loss(x, x_lens, y)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss, on_epoch=True, on_step=True)
        return result

    def validation_step(self, batch, batch_idx):
        x, x_lens, y, _ = batch
        loss = self.model.loss(x, x_lens, y)
        y_pred = self.model.decode(x, x_lens)
        return {"val_loss": loss, "y": y, "y_pred": y_pred}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = []
        y_pred = []
        for output in outputs:
            for instance in output["y"].tolist():
                y.append([(tag, i) for i, tag in enumerate(instance)])
            for instance in output["y_pred"].tolist():
                y_pred.append([(tag, i) for i, tag in enumerate(instance)])

        micro_f1 = self._score(y, y_pred)

        logs = {"val_loss": avg_loss, "val_micro_f1": micro_f1}
        return {"val_loss": avg_loss, "log": logs, "progress_bar": logs}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        result = self.validation_epoch_end(outputs)
        logs = {
            "test_loss": result["log"]["val_loss"],
            "test_micro_f1": result["log"]["val_micro_f1"],
        }
        return {
            "test_loss": result["val_loss"],
            "log": logs,
            "progress_bar": logs,
        }

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.model.hparams.train.adam.lr,
            weight_decay=self.model.hparams.train.adam.weight_decay,
        )

    def _score(self, y_true, y_pred):
        metric = Metric("micro avg")
        metric.score(
            y_true,
            y_pred,
            ignore_classes=[
                self.label_vocab.get_id(PAD_TOKEN),
                self.label_vocab.get_id(O_TOKEN),
            ],
        )
        return metric.micro_avg_f_score()