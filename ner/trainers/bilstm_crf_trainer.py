"/home/zoro/projects/data/entity-recognition-datasets/data/conll2003/CONLL-format/data"

import sys
import os
import logging
from ner.utils.gpu_profile import gpu_profile

import pytorch_lightning as pl
from ner.data.conll03 import CoNLL2003Dataset, create_vocab
from ner.models.bilstm_crf import BiLSTMCRF
from ner.utils.samplers import BucketBatchSampler, BatchSampler
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import EarlyStopping

print(os.environ.get("COMET_API_KEY"))
comet_logger = CometLogger(
    api_key=os.environ.get("COMET_API_KEY"),
    workspace=os.environ.get("COMET_WORKSPACE"),
    project_name="ner",
)


logging.getLogger("lightning").setLevel(logging.DEBUG)

train_fpath = "/home/zoro/projects/data/entity-recognition-datasets/data/conll2003/CONLL-format/data/eng.train"
val_fpath = "/home/zoro/projects/data/entity-recognition-datasets/data/conll2003/CONLL-format/data/eng.testa"

word_vocab_fpath = "store/word_vocab.txt"
char_vocab_fpath = "store/char_vocab.txt"
ner_vocab_fpath = "store/ner_vocab.txt"


def main():

    word_vocab, ner_vocab, char_vocab = create_vocab(
        train_fpath, word_vocab_fpath, ner_vocab_fpath, char_vocab_fpath
    )

    train_dataset = CoNLL2003Dataset(train_fpath, word_vocab, ner_vocab, char_vocab)

    # print(
    #     len(train_dataset.sentences),
    #     len(train_dataset.labels),
    #     len(train_dataset.char_sequences),
    # )

    val_dataset = CoNLL2003Dataset(val_fpath, word_vocab, ner_vocab, char_vocab)
    # print(
    #     len(val_dataset.sentences), len(val_dataset.labels), len(val_dataset.char_sequences),
    # )

    # print(
    #     len(test_dataset.sentences),
    #     len(test_dataset.labels),
    #     len(test_dataset.char_sequences),
    # )

    # print(train_dataset.sentences[0])
    # print(" ".join([word_vocab.get_token(i) for i in train_dataset.sentences[0]]))
    # print(train_dataset.labels[0])
    # print(train_dataset.char_sequences[0])

    train_sampler = BucketBatchSampler(
        data_source=train_dataset,
        bucket_boundaries=[5, 10, 15, 20, 25, 30, 40, 50, 140],
        seq_len_fn=CoNLL2003Dataset.seq_len_fn,
        batch_size=32,
        shuffle=True,
    )

    val_sampler = BucketBatchSampler(
        data_source=val_dataset,
        bucket_boundaries=[5, 10, 15, 20, 25, 30, 40, 50, 140],
        seq_len_fn=CoNLL2003Dataset.seq_len_fn,
        batch_size=32,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=CoNLL2003Dataset.collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=32, collate_fn=CoNLL2003Dataset.collate_fn,
    )

    # seq_lens = []
    # for i in range(2):
    #     seq_lens.append(len(train_dataset.sentences[i]))

    # print(seq_lens)
    # print(max(seq_lens))

    # for x in train_loader:
    #     words_t, tags_t, chars_t = x
    #     print(words_t.shape)
    #     print(tags_t.shape)
    #     print(chars_t.shape)
    #     print("-" * 50)

    # """
    model = BiLSTMCRF()

    # pl.seed_everything(42)

    # default early stopping criterion
    early_stop = EarlyStopping(
        monitor="val_loss", patience=6, strict=False, verbose=False, mode="min"
    )

    trainer = pl.Trainer(
        gpus=1,
        # precision=16,
        # logger=comet_logger,
        # fast_dev_run=True,  # default is false
        # deterministic=True,  # default is True. For reproducibility
        # auto_lr_find=True,  # default is false. needs lr defined in hparams
        check_val_every_n_epoch=1,  # default is 1
        early_stop_callback=False,  # default is None. If true, uses default callback or we pass early_stop,
        # track_grad_norm=2,
        # overfit_batches=0.01,
        weights_summary="full",
        log_gpu_memory="all"
        # auto_scale_batch_size=True,
    )
    trainer.fit(model, train_loader, val_dataloader)

    trainer.test(test_dataloaders=val_dataloader)


# """
if __name__ == "__main__":
    sys.settrace(gpu_profile)

    main()
