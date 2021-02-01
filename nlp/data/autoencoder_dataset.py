import os
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from collections import Counter
from nlp.data.vocab import Vocab
from nlp.constants import BOS_TOKEN, EOS_TOKEN
from pathlib import Path

class AutoencoderDataset(Dataset):
    
    def __init__(
        self,
        tokens_list: List[List[str]],
        vocab_fpath: os.PathLike,
        max_vocab_size: int = 100_000,
        max_sequence_length: int = 256,
        use_bos: bool = True,
    ):
        super(AutoencoderDataset, self).__init__()
        
        self.tokens_list = tokens_list
        self.vocab_fpath = vocab_fpath
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.use_bos = use_bos
        self.__create_vocab()
        
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_outputs = []
        self.__process()
        
    def __getitem__(self, i):
        return self.encoder_inputs[i], self.decoder_inputs[i], self.decoder_outputs[i]
    
    def __len__(self):
        return len(self.encoder_inputs)
    
    @classmethod
    def seq_len_fn(cls, item):
        return len(item[0])
    
    def __process(self):
        for tokens in self.tokens_list:
            encoder_inputs = tokens
            if self.use_bos:
                decoder_inputs = [BOS_TOKEN] + tokens
                decoder_outputs = tokens + [EOS_TOKEN]
            else:
                decoder_inputs = tokens
                decoder_outputs = tokens[1:] + [EOS_TOKEN]            
            self.encoder_inputs.append([self.vocab.get_id(tok) for tok in encoder_inputs])
            self.decoder_inputs.append([self.vocab.get_id(tok) for tok in decoder_inputs])
            self.decoder_outputs.append([self.vocab.get_id(tok) for tok in decoder_outputs])
            
    def get_vocab_len(self):
        return len(self.vocab)
        
    @classmethod
    def collate_fn(cls, batch):
        batch_encoder_inputs = []
        batch_decoder_inputs, batch_decoder_outputs = [], []
        batch_encoder_lens, batch_decoder_lens = [], []
        batch = sorted(batch, key=lambda x: -len(x[0]))
        for encoder_inputs, decoder_inputs, decoder_outputs in batch:
            batch_encoder_inputs.append(torch.tensor(encoder_inputs, dtype=torch.long))
            batch_decoder_inputs.append(torch.tensor(decoder_inputs, dtype=torch.long))
            batch_decoder_outputs.append(torch.tensor(decoder_outputs, dtype=torch.long))
            batch_encoder_lens.append(len(encoder_inputs))
            batch_decoder_lens.append(len(decoder_inputs))
            
        batch_encoder_inputs = pad_sequence(batch_encoder_inputs, batch_first=True)
        batch_decoder_inputs = pad_sequence(batch_decoder_inputs, batch_first=True)
        batch_decoder_outputs = pad_sequence(batch_decoder_outputs, batch_first=True)
        batch_encoder_lens = torch.tensor(batch_encoder_lens, dtype=torch.long)
        batch_decoder_lens = torch.tensor(batch_decoder_lens, dtype=torch.long)
        
        return batch_encoder_inputs, batch_decoder_inputs, batch_decoder_outputs, batch_encoder_lens, batch_decoder_lens
        
    def __create_vocab(self):
        self.vocab = Vocab(set_unk=True)
        try:
            self.vocab.load(self.vocab_fpath)
        except FileNotFoundError:
            print("creating vocab...")
            if self.use_bos:
                self.vocab.add(BOS_TOKEN)
            self.vocab.add(EOS_TOKEN)
            
            counter = Counter([tok for tokens in self.tokens_list for tok in tokens])
            for tok,_ in counter.most_common(self.max_vocab_size):
                self.vocab.add(tok)
                
            Path(self.vocab_fpath).parent.mkdir(parents=True, exist_ok=True)
            self.vocab.save(self.vocab_fpath)
            
def test_autoencoder_dataset():
    cwd = os.getcwd()
    texts = [
        "rose is red",
        "grass is green",
        "rose is not green",
        "grass is red not",
    ]
    tokens_list = [text.split(" ") for text in texts]
    
    vocab_fpath = os.path.join(cwd, "vocab.yaml")
    max_vocab_size = 6
    use_bos = True
    max_sequence_length = 5
    
    if os.path.exists(vocab_fpath):
        os.remove(vocab_fpath)
        
    dtst = AutoencoderDataset(
        tokens_list,
        vocab_fpath,
        max_vocab_size,
        max_sequence_length,
        use_bos,
    )
    
    assert os.path.exists(vocab_fpath)
    assert dtst.vocab.get_id("[PAD]") == 0
    assert dtst.vocab.get_id("[UNK]") == 1
    if not use_bos:
        assert dtst.vocab.get_id("[EOS]") == 2
        assert "[BOS]" not in dtst.vocab.tok2id
    else:
        assert dtst.vocab.get_id("[BOS]") == 2
        assert dtst.vocab.get_id("[EOS]") == 3
    
    os.remove(vocab_fpath)
    
if __name__=="__main__":
    test_autoencoder_dataset()