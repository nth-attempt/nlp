from ner.utils.samplers import BucketBatchSampler
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np
import pytorch_lightning as pl

pl.seed_everything(42)


class MyDataset(Dataset):
    def __init__(self):
        alphabets = [x for x in range(ord("a"), ord("z") + 1)]
        num_train = 10
        self.data = [
            torch.tensor(
                np.random.choice(
                    alphabets, int(np.random.randint(2, 33, 1)[0])
                )
            )
            for i in range(num_train)
        ]
        print("lengths", [len(x) for x in self.data])

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    return (
        pad_sequence(batch, batch_first=True),
        torch.tensor([len(x) for x in batch]),
    )


def seq_len_fn(row):
    return len(row)


batch_size = 8
dataset = MyDataset()
batch_sampler = BucketBatchSampler(
    dataset,
    seq_len_fn=seq_len_fn,
    bucket_boundaries=[8, 16, 32, 48],
    batch_size=batch_size,
    shuffle=False,
)

dataloader = DataLoader(
    dataset, batch_sampler=batch_sampler, collate_fn=collate_fn
)
default_dataloader = DataLoader(
    dataset, batch_size=batch_size, collate_fn=collate_fn
)

for i in range(3):
    print("epoch", i)
    for x, x_lens in dataloader:
        print("batch size and lengths", len(x), x_lens)

print("-" * 50, "Defaults")
for i in range(2):
    print("epoch", i)
    for x, x_lens in default_dataloader:
        print("batch size and lengths", len(x), x_lens)

