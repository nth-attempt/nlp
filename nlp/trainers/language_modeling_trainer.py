import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from nlp.data.language_model_dataset import LanguageModelDataset
from nlp.models.language_modeling.rnn_language_model import RNNLanguageModel
from nlp.samplers import BucketBatchSampler

cwd = os.getcwd()
logs_dir = os.path.join(os.path.dirname(cwd), "lightning_logs")

logger = TensorBoardLogger(logs_dir, name="language_model", default_hp_metric=False)

def main():
    
    pl.seed_everything(42)
    conf = OmegaConf.load("config/rnn_language_model.yaml")
    
    train_tokens_list, val_tokens_list = create_dummy_data()
    
    train_dataset = LanguageModelDataset(
        train_tokens_list,
        conf.data.vocab_fpath,
        conf.data.max_vocab_size,
        conf.data.max_sequence_length,
        conf.data.use_bos,
    )    
    val_dataset = LanguageModelDataset(
        val_tokens_list,
        conf.data.vocab_fpath,
        conf.data.max_vocab_size,
        conf.data.max_sequence_length,
        conf.data.use_bos,
    )
    
    train_sampler = BucketBatchSampler(
        data_source=train_dataset,
        bucket_boundaries=conf.data.bucket_boundaries,
        seq_len_fn=LanguageModelDataset.seq_len_fn,
        batch_size=conf.train.batch_size,
        shuffle=True,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=LanguageModelDataset.collate_fn,
        num_workers=32,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=conf.train.batch_size,
        collate_fn=LanguageModelDataset.collate_fn,
        shuffle=False,
        num_workers=32,
    )
    
    conf.data.input_vocab_size = train_dataset.get_vocab_len()
    conf.data.output_vocab_size = conf.data.input_vocab_size
    
    model = RNNLanguageModel(conf)
    
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
    num_train, num_val = 3, 1
    tokens = ["a", "b", "c", "d", "a", "b", "c", "d"]
    train_tokens = [tokens]*num_train
    val_tokens = [tokens]*num_val
    return train_tokens, val_tokens

def save_and_load_encoder():
    from nlp.modules.seq2seq_encoders import RNNSeq2SeqEncoder
    from nlp.modules.embedding import Embedding
    # save best model encoder and embedding
    lm = RNNLanguageModel.load_from_checkpoint(best_model_path)
    lm.save_embedding() # saves in conf.model.embedding.save_torch_filepath
    lm.save_encoder() # saves in conf.model.encoder.save_filepath
    
    emb = Embedding(vocab_size, conf.model.embedding.embedding_size, filepath=conf.model.embedding.save_torch_filepath)
    emb.eval()
    enc = RNNSeq2SeqEncoder(conf.model.embedding.embedding_size, conf.model.encoder.hidden_size, bidirectional=False)
    enc.load_torch(conf.model.encoder.save_filepath)
    enc.eval()
    
    input_l = [[1,2,3,4], [5,6,7,8]]
    x = torch.tensor(input_l)
    lens = torch.tensor([len(item) for item in input_l])
    # use hidden for sentence representations
    x_emb = emb(x)
    enc_emissions, hiddens = enc(x_emb, lens)
    

if __name__=="__main__":
    main()