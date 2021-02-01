import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from nlp.data.autoencoder_dataset import AutoencoderDataset
from nlp.models.autoencoder.rnn_autoencoder import RNNAutoencoder
from nlp.samplers import BucketBatchSampler

cwd = os.getcwd()
logs_dir = os.path.join(os.path.dirname(cwd), "lightning_logs")

logger = TensorBoardLogger(logs_dir, name="autoencoder", default_hp_metric=False)

def main():
    pl.seed_everything(42)
    conf = OmegaConf.load("config/rnn_autoencoder.yaml")
    
    train_tokens_list, val_tokens_list = create_dummy_data()
    
    train_dataset = AutoencoderDataset(
        train_tokens_list,
        conf.data.vocab_fpath,
        conf.data.max_vocab_size,
        conf.data.max_sequence_length,
        conf.data.use_bos,
    )    
    val_dataset = AutoencoderDataset(
        val_tokens_list,
        conf.data.vocab_fpath,
        conf.data.max_vocab_size,
        conf.data.max_sequence_length,
        conf.data.use_bos,
    )
    
    train_sampler = BucketBatchSampler(
        data_source=train_dataset,
        bucket_boundaries=conf.data.bucket_boundaries,
        seq_len_fn=AutoencoderDataset.seq_len_fn,
        batch_size=conf.train.batch_size,
        shuffle=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=AutoencoderDataset.collate_fn,
        num_workers=32,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.train.batch_size,
        collate_fn=AutoencoderDataset.collate_fn,
        shuffle=False,
        num_workers=32,
    )
    
    conf.data.input_vocab_size = train_dataset.get_vocab_len()
    conf.data.output_vocab_size = conf.data.input_vocab_size
    
    model = RNNAutoencoder(conf)
    
    checkpoint_callback = ModelCheckpoint(
        filepath=logs_dir,
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=3,
        mode="min",
    )
    
    trainer = pl.Trainer(
        max_epochs=20,
        logger=logger,
        callbacks=[early_stopping],
        gpus=0,
        weights_summary="full",
    )
    
    trainer.fit(model, train_dataloader, val_dataloader)
    print(checkpoint_callback.best_model_path)
    
def create_dummy_data():
    num_train, num_val = 5, 1
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h"]
    train_tokens = [tokens]*num_train
    val_tokens = [tokens]*num_val
    return train_tokens, val_tokens

if __name__=="__main__":
    main()

