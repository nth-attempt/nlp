import pytorch_lightning as pl
from nlp.data.conll2003 import CoNLL2003Dataset
from nlp.models.sequence_labeling import BiRNN
from nlp.samplers import BucketBatchSampler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from pytorch_lightning.loggers import CometLogger
# # print(os.environ.get("COMET_API_KEY"))
# comet_logger = CometLogger(
#     api_key=os.environ.get("COMET_API_KEY"),
#     workspace=os.environ.get("COMET_WORKSPACE"),
#     project_name="nlp",
# )


def main():
    pl.seed_everything(42)

    conf = OmegaConf.load("config/bilstm_crf.yaml")

    train_dataset = CoNLL2003Dataset(
        conf.data.train_file,
        conf.data.word_vocab_file,
        conf.data.label_vocab_file,
        conf.data.char_vocab_file,
    )
    val_dataset = CoNLL2003Dataset(
        conf.data.val_file,
        conf.data.word_vocab_file,
        conf.data.label_vocab_file,
        conf.data.char_vocab_file,
    )
    test_dataset = CoNLL2003Dataset(
        conf.data.test_file,
        conf.data.word_vocab_file,
        conf.data.label_vocab_file,
        conf.data.char_vocab_file,
    )

    train_sampler = BucketBatchSampler(
        data_source=train_dataset,
        bucket_boundaries=[5, 10, 15, 20, 25, 30, 40, 50, 140],
        seq_len_fn=CoNLL2003Dataset.seq_len_fn,
        batch_size=conf.train.batch_size,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=CoNLL2003Dataset.collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.train.batch_size,
        collate_fn=CoNLL2003Dataset.collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=conf.train.batch_size,
        collate_fn=CoNLL2003Dataset.collate_fn,
    )

    (
        word_vocab_len,
        label_vocab_len,
        char_vocab_len,
    ) = train_dataset.get_vocab_lens()

    conf.model.num_words = word_vocab_len
    conf.model.num_labels = label_vocab_len

    model = BiRNN(conf, label_vocab=train_dataset.label_vocab)

    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_loss")],
        gpus=1,
        # precision=16,
        # logger=comet_logger,
        fast_dev_run=False,  # default is false
        # overfit_batches=0.01,
        weights_summary="full",
        # log_gpu_memory="all"
        # auto_scale_batch_size=True,
    )
    trainer.fit(model, train_loader, val_dataloader)

    trainer.test(test_dataloaders=test_dataloader)


if __name__ == "__main__":
    main()
