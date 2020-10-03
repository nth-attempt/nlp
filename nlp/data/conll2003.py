# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common helper functions for preprocessing Named Entity Recognition (NER) datasets."""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nlp.data.vocab import Vocab
import os
from collections import Counter
from pathlib import Path


class CoNLL2003Dataset(Dataset):
    """Some Information about CoNLL2003Dataset"""

    def __init__(
        self,
        file_path: os.PathLike,
        word_vocab_fpath: os.PathLike,
        label_vocab_fpath: os.PathLike,
        char_vocab_fpath: os.PathLike,
        max_word_len: int = 256,  # TODO: unused for now
        max_char_len: int = 100,  # TODO: unused for now
    ):
        super(CoNLL2003Dataset, self).__init__()

        with open(file_path) as f:
            self.data = f.read()
        (
            self.word_vocab,
            self.label_vocab,
            self.char_vocab,
        ) = CoNLL2003Dataset.create_vocab(
            file_path, word_vocab_fpath, label_vocab_fpath, char_vocab_fpath
        )

        self.sentences = []
        self.labels = []
        self.char_sequences = []

        self.preprocess()

    def __getitem__(self, i):

        return self.sentences[i], self.labels[i], self.char_sequences[i]

    def __len__(self):
        return len(self.sentences)

    def get_vocab_lens(self):
        return (
            len(self.word_vocab),
            len(self.label_vocab),
            len(self.char_vocab),
        )

    @classmethod
    def collate_fn(cls, batch):
        max_char_len = 0
        sents, sent_lens, labels, char_seqs = [], [], [], []
        batch = sorted(batch, key=lambda x: -cls.seq_len_fn(x))
        for sent, lbl, chr_seq in batch:
            sents.append(torch.tensor(sent, dtype=torch.long))
            sent_lens.append(len(sent))
            labels.append(torch.tensor(lbl, dtype=torch.long))
            max_char_len = max(
                max_char_len, max(len(chrs) for chrs in chr_seq)
            )
            char_seqs.append(chr_seq)

        sents_t = pad_sequence(sents, batch_first=True)
        labels_t = pad_sequence(labels, batch_first=True)
        sent_lens_t = torch.tensor(sent_lens, dtype=torch.long)
        char_seqs_t = torch.zeros(
            sents_t.shape[0], sents_t.shape[1], max_char_len
        )
        for i, chr_seq in enumerate(char_seqs):
            for c, chars in enumerate(chr_seq):
                char_seqs_t[i, c, : len(chars)] = torch.tensor(
                    chars, dtype=torch.long
                )
        # TODO: add char lengths
        return sents_t, sent_lens_t, labels_t, char_seqs_t

    @classmethod
    def seq_len_fn(cls, instance):
        return len(instance[0])

    def preprocess(self):
        text_list = self.data.split("\n\n")

        for doc in text_list:
            lines = doc.split("\n")
            words, ner_tags, chars = [], [], []
            for line in lines:
                splits = line.split()
                if len(splits) > 1 and splits[0] != "-DOCSTART-":
                    words.append(self.word_vocab.get_id(splits[0]))
                    ner_tags.append(self.label_vocab.get_id(splits[-1]))
                    chars.append(
                        [self.char_vocab.get_id(ch) for ch in splits[0]]
                    )
            if words and ner_tags:
                self.sentences.append(words)
                self.labels.append(ner_tags)
                self.char_sequences.append(chars)

        return self.sentences, self.labels, self.char_sequences

    @staticmethod
    def create_vocab(
        train_fpath: os.PathLike,
        word_vocab_fpath: os.PathLike,
        label_vocab_fpath: os.PathLike,
        char_vocab_fpath: os.PathLike,
        max_vocab_size: int = 100_000,
    ):
        word_vocab = Vocab(set_unk=True)
        label_vocab = Vocab()
        char_vocab = Vocab(set_unk=True)
        try:
            # load vocabs from file if exsist
            print("Loading vocabs from file")
            word_vocab.load(word_vocab_fpath)
            label_vocab.load(label_vocab_fpath)
            char_vocab.load(char_vocab_fpath)
        except FileNotFoundError:
            # build vocab
            print("Creating vocabs")
            word_counter = Counter()
            with open(train_fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    splits = line.split()
                    if len(splits) > 1 and splits[0] != "-DOCSTART-":
                        for char in splits[0]:
                            char_vocab.add(char)
                        word_counter.update([splits[0]])
                        label_vocab.add(splits[-1])

            for word, _ in word_counter.most_common(max_vocab_size):
                word_vocab.add(word)

            Path(word_vocab_fpath).parent.mkdir(parents=True, exist_ok=True)
            Path(label_vocab_fpath).parent.mkdir(parents=True, exist_ok=True)
            Path(char_vocab_fpath).parent.mkdir(parents=True, exist_ok=True)

            # save
            word_vocab.save(word_vocab_fpath)
            label_vocab.save(label_vocab_fpath)
            char_vocab.save(char_vocab_fpath)

        return word_vocab, label_vocab, char_vocab
