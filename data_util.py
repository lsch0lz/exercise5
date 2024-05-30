from collections import UserDict
from typing import List, Optional, Tuple

import torch
from torch import LongTensor, Tensor

Sentence = List[str]


def load_corpus(
        path, *, character_level, truncate_length: Optional[int]
) -> List[Sentence]:
    sentences = []
    with open(path) as text_file:
        for line in text_file.readlines():
            if character_level:
                tokens = [char for char in line]
            else:
                # Default is loading the corpus as sequence of words
                tokens = line.lower().strip().split()
            if truncate_length is not None:
                tokens = tokens[:truncate_length]
            sentences.append(tokens)
    return sentences


def split_corpus(
        corpus: List[Sentence], split_sizes: Tuple[float, ...] = (0.8, 0.1, 0.1)
) -> Tuple[List[Sentence], ...]:
    """Split the corpus into parts based on the specified sizes."""

    if not all(s >= 0 for s in split_sizes):
        msg = "All split sizes need to be 0 or larger."
        raise ValueError(msg)

    if not sum(split_sizes) > 0:
        msg = "The sum of the split sizes must be larger than 0."
        raise ValueError(msg)

    # Make sure they sum up to 1
    split_sizes = tuple(s / sum(split_sizes) for s in split_sizes)

    splits = ()

    start_index = 0
    for s in split_sizes:
        end_index = start_index + len(corpus) * s
        splits += (corpus[round(start_index): round(end_index)],)
        start_index = end_index

    return splits


class Vocabulary(UserDict):
    UNK_TOKEN: str = "<UNK>"
    START_TOKEN: str = "<START>"
    STOP_TOKEN: str = "<STOP>"
    PAD_TOKEN: str = "<PAD>"

    @classmethod
    def from_data(
            cls, sentences: List[Sentence], unk_threshold: int = 0
    ) -> "Vocabulary":
        """
        Make a dictionary of words given a list of tokenized sentences.

        :param data: List of (sentence, label) tuples
        :param unk_threshold: All tokens below this count threshold are excluded from dictionary and replaced with UNK
        :return: A dictionary of string keys and index values
        """

        token_frequencies = {}
        for sent in sentences:
            for token in sent:
                if token not in token_frequencies:
                    token_frequencies[token] = 0
                token_frequencies[token] += 1

        # Assign indices to each distinct ngram
        token_to_index = {
            cls.PAD_TOKEN: 0,
            cls.START_TOKEN: 1,
            cls.STOP_TOKEN: 2,
            cls.UNK_TOKEN: 3,
        }

        for word, freq in token_frequencies.items():
            if freq > unk_threshold:  # only add words that are above threshold
                token_to_index[word] = len(token_to_index)

        return cls(token_to_index)

    def make_index_vectors(
            self,
            sentences: List[Sentence],
            *,
            as_targets: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Make the index vectors from the provided sentence and return them with a sequence length tensor.

        All sentences should be right-padded to the same length using the padding token.
        A <START> token should be prepended unless the as_targets argument is set to true.
        A <STOP> token should be appended to the end of the sequence if the as_targets token is set.
        """
        # add 1 for either <START> or <STOP> Token
        max_sentence_len: int = max(len(sentence) for sentence in sentences) + 1

        batch_size: int = len(sentences)
        padded_vectors: Tensor = torch.full((batch_size, max_sentence_len), self[self.PAD_TOKEN])
        sequence_lengths: Tensor = torch.zeros(batch_size)

        for idx, sentence in enumerate(sentences):
            indices: List = [self.get(token, self[self.UNK_TOKEN]) for token in sentence]

            if not as_targets:
                indices: List = [self[self.START_TOKEN]] + indices
            if as_targets:
                indices.append(self[self.STOP_TOKEN])

            padded_vectors[idx, :len(indices)] = torch.tensor(indices)
            sequence_lengths[idx] = len(indices)

        return padded_vectors, sequence_lengths
