from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from data_util import Vocabulary


class LanguageModel(torch.nn.Module):  # inherits from nn.Module

    def __init__(
            self,
            vocab_size: int,
            embedding_size: int,
            rnn_hidden_size: int,
            pad_index: int,
            num_layers: int = 1,
            is_character_level: bool = False,
    ):
        super(LanguageModel, self).__init__()

        # Remember device, vocabulary and character-level
        self.device = None
        self.is_character_level = is_character_level
        self.vocab_size = vocab_size
        self.pad_index = pad_index

        # Embeddings of words (or characters)
        self.embedding = torch.nn.Embedding(
            vocab_size, embedding_size, padding_idx=pad_index
        )

        # The LSTM takes embeddings as inputs, and outputs hidden states of dimensionality hidden dim
        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # The hidden2tag linear layer takes LSTM output and projects to tag space
        self.hidden2tag = torch.nn.Linear(rnn_hidden_size, vocab_size)

        # TODO: Maybe delete ignore_index
        self.loss_function = torch.nn.NLLLoss(ignore_index=self.pad_index)

    def to(self, device):
        self.device = device
        super().to(device)

    def forward(
            self,
            input_ids: torch.LongTensor,
            input_lengths: torch.LongTensor,
            *,
            target_ids: Optional[torch.LongTensor] = None,
            hidden_state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Compute the forward pass through the model.

        If the target_ids are given compute the loss as well.

        :param input_ids: A list of sentences (each represented by a list of token ids) [batch size x sequence_length]
        :param target_ids: A list of sentences (each represented by a list of token ids) [batch size x sequence_length]
        :return: Dictionary of computed values (including the loss).
        """

        outputs = {}
        embeddings: Tensor = self.embedding(input_ids)

        packed_input: PackedSequence = pack_padded_sequence(embeddings, input_lengths, batch_first=True, enforce_sorted=False)

        packed_output, hidden_state = self.lstm(packed_input, hidden_state)

        seq_unpacked, lens_unpacked = pad_packed_sequence(packed_output, batch_first=True)

        features: Tensor = self.hidden2tag(seq_unpacked)

        log_probs: Tensor = F.log_softmax(features, dim=-1)

        outputs["log_softmax"] = log_probs
        outputs["last_hidden"] = hidden_state

        if target_ids is not None:
            loss = self.loss_function(log_probs.view(-1, self.vocab_size), target_ids.view(-1))
            outputs["loss"] = loss


        return outputs

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def generate_text(
            self,
            vocabulary: Vocabulary,
            prefix: Optional[str] = None,
            max_symbols: int = 100,
            temperature=1.0,
    ):
        if not prefix:
            prefix_tokens = []
        elif self.is_character_level:
            prefix_tokens = list(prefix)
        else:
            prefix_tokens = prefix.split()

        # see: https://moodle.hu-berlin.de/mod/forum/discuss.php?d=1008359
        # prefix_tokens.insert(0, vocabulary[vocabulary.START_TOKEN])

        inv_map = {v: k for k, v in vocabulary.items()}

        input_ids, input_lengths = vocabulary.make_index_vectors([prefix_tokens])
        input_ids = input_ids.to(self.device)

        hidden_state = None
        generated_tokens = []

        for _ in range(max_symbols):
            output = self.forward(input_ids, input_lengths, hidden_state=hidden_state)

            # only lookt at the first item in the batch, the last element in the sequence
            token_weights = (
                output["log_softmax"][0, -1, :].squeeze().div(temperature).exp().cpu()
            )

            assert token_weights.shape == (self.vocab_size,)

            # Do not predict certain tokens
            token_weights[vocabulary[vocabulary.UNK_TOKEN]] = 0
            token_weights[vocabulary[vocabulary.PAD_TOKEN]] = 0
            token_weights[vocabulary[vocabulary.START_TOKEN]] = 0

            token_id = torch.multinomial(token_weights, 1)[0]

            if token_id == vocabulary[vocabulary.STOP_TOKEN]:
                break

            input_ids = torch.tensor([[token_id]], device=self.device)
            input_lengths = torch.tensor([1])

            generated_tokens.append(inv_map[token_id.item()])
            hidden_state = output["last_hidden"]

        separator = "" if self.is_character_level else " "

        if prefix:
            print(f"'{prefix}' continues as '{separator.join(generated_tokens)}'")
        else:
            print(f"Generating sample text: {separator.join(generated_tokens)}")

        return separator.join(generated_tokens)
