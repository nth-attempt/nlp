"/home/zoro/projects/data/entity-recognition-datasets/data/conll2003/CONLL-format/data"

import sys
import os
import logging

import pytorch_lightning as pl
from ner.data.conll2003 import CoNLL2003Dataset, create_vocab
from ner.models.bilstm_crf import BiLSTMCRF
from ner.utils.samplers import BucketBatchSampler, BatchSampler
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import EarlyStopping
from omegaconf import OmegaConf

print(os.environ.get("COMET_API_KEY"))
comet_logger = CometLogger(
    api_key=os.environ.get("COMET_API_KEY"),
    workspace=os.environ.get("COMET_WORKSPACE"),
    project_name="ner",
)


def main():
    pl.seed_everything(42)

    conf = OmegaConf.load("config/bilstm_crf.yaml")

    word_vocab, ner_vocab, char_vocab = create_vocab(
        conf.data.train_file,
        conf.data.word_vocab_file,
        conf.data.ner_vocab_file,
        conf.data.char_vocab_file,
    )

    train_dataset = CoNLL2003Dataset(
        conf.data.train_file, word_vocab, ner_vocab, char_vocab
    )

    # print(
    #     len(train_dataset.sentences),
    #     len(train_dataset.labels),
    #     len(train_dataset.char_sequences),
    # )

    val_dataset = CoNLL2003Dataset(
        conf.data.val_file, word_vocab, ner_vocab, char_vocab
    )
    # print(
    #     len(val_dataset.sentences),
    #     len(val_dataset.labels),
    #     len(val_dataset.char_sequences),
    # )

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

    model = BiLSTMCRF(conf)

    trainer = pl.Trainer(
        gpus=1,
        # precision=16,
        # logger=comet_logger,
        # fast_dev_run=True,  # default is false
        # deterministic=True,  # default is True. For reproducibility
        # auto_lr_find=True,  # default is false. needs lr defined in hparams
        check_val_every_n_epoch=1,  # default is 1
        early_stop_callback=True,  # default is None. If true, uses default callback or we pass early_stop,
        # track_grad_norm=2,
        # overfit_batches=0.01,
        weights_summary="full",
        # log_gpu_memory="all"
        # auto_scale_batch_size=True,
    )
    trainer.fit(model, train_loader, val_dataloader)


if __name__ == "__main__":
    main()
