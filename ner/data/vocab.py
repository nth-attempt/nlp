from pathlib import Path
import os
from ner.utils.constants import UNK, UNK_TOKEN, PAD_TOKEN


class Vocab:
    def __init__(self, set_unk=False):
        self.tok2id = {}
        self.id2tok = {}
        self.set_unk = set_unk
        self.next_id = 0

        self.add(PAD_TOKEN)
        if set_unk:
            self.add(UNK_TOKEN)

    def add(self, tok: str):
        if tok not in self.tok2id:
            self.tok2id[tok] = self.next_id
            self.id2tok[self.next_id] = tok
            self.next_id += 1

    def get_id(self, tok: str):
        try:
            return self.tok2id[tok]
        except KeyError:
            if self.set_unk:
                return UNK
            else:
                raise KeyError(f"Unknown Token: {tok}")

    def __len__(self):
        return len(self.tok2id)

    def get_token(self, id: int):
        try:
            return self.id2tok[id]
        except KeyError:
            raise KeyError(f"Unknown Index: {id}")

    def load(self, vocab_file: os.PathLike):
        vocab_file = Path(vocab_file)
        try:
            with open(vocab_file) as f:
                for id, tok in enumerate(f.readlines()):
                    tok = tok[:-1]
                    self.tok2id[tok] = id
                    self.id2tok[id] = tok
        except FileNotFoundError:
            raise FileNotFoundError(f"{vocab_file} not found")

    def save(self, vocab_file: os.PathLike):
        vocab_file = Path(vocab_file)
        if vocab_file.exists():
            print(f"Overwriting the exisitng vocab file: {vocab_file}")

        with open(vocab_file, "w", encoding="utf-8") as f:
            for i in range(len(self.id2tok)):
                f.write(f"{self.id2tok[i]}\n")
