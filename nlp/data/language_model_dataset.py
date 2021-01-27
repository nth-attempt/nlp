import os
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from collections import Counter
from nlp.data.vocab import Vocab
from nlp.constants import BOS_TOKEN, EOS_TOKEN
from pathlib import Path

class LanguageModelDataset(Dataset):
    
    def __init__(
        self,
        tokens_list: List[List[str]],
        vocab_fpath: os.PathLike,
        max_vocab_size: int = 100_000,
        max_sequence_length: int = 256,
        use_bos: bool = True,
    ):
        super(LanguageModelDataset, self).__init__()
        
        self.tokens_list = tokens_list
        self.vocab_fpath = vocab_fpath
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.use_bos = use_bos
        self.__create_vocab()
        
        self.inputs = []
        self.outputs = []
        self.__process()
        
    def __getitem__(self, i):
        return self.inputs[i], self.outputs[i]
    
    def __len__(self):
        return len(self.inputs)
    
    @classmethod
    def seq_len_fn(cls, instance):
        return len(instance[0])
    
    def __process(self):
        
        for tokens in self.tokens_list:
            if self.use_bos:
                inputs = [BOS_TOKEN] + tokens
                outputs = inputs + [EOS_TOKEN]
            else:
                inputs = tokens
                outputs = tokens[1:] + [EOS_TOKEN]
            self.inputs.append([self.vocab.get_id(tok) for tok in inputs])
            self.outputs.append([self.vocab.get_id(tok) for tok in outputs])
            
    def get_vocab_len(self):
        return len(self.vocab)
    
    @classmethod
    def collate_fn(cls, batch):
        batch_inputs, batch_input_lens, batch_outputs = [], [], []
        batch = sorted(batch, key=lambda x: -len(x[0]))
        for inputs, outputs in batch:
            batch_inputs.append(torch.tensor(inputs, dtype=torch.long))
            batch_input_lens.append(len(inputs))
            batch_outputs.append(torch.tensor(outputs, dtype=torch.long))
        
        batch_inputs = pad_sequence(batch_inputs, batch_first=True)
        batch_input_lens = torch.tensor(batch_input_lens, dtype=torch.long)
        batch_outputs = pad_sequence(batch_outputs, batch_first=True)
        
        return batch_inputs, batch_input_lens, batch_outputs
    
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
            
def test_language_model_dataset():
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
    use_bos = False
    max_sequence_length = 5
    
    if os.path.exists(vocab_fpath):
        os.remove(vocab_fpath)
        
    dtst = LanguageModelDataset(
        tokens_list,
        vocab_fpath,
        max_vocab_size,
        max_sequence_length,
        use_bos,
    )
    
    assert os.path.exists(vocab_fpath)
    assert dtst.vocab.get_id("[PAD]") == 0
    assert dtst.vocab.get_id("[UNK]") == 1
    assert dtst.vocab.get_id("[EOS]") == 2
    assert "[BOS]" not in dtst.vocab.tok2id
    
    os.remove(vocab_fpath)
    
if __name__=="__main__":
    test_language_model_dataset()
    
    