import torch

from data_util import Vocabulary, split_corpus


def test_input_vectors():
    training_data = [
        "this is a good movie".split(" "),
        "it is raining today".split(" "),
    ]

    vocabulary = Vocabulary.from_data(training_data)

    vectors, lengths = vocabulary.make_index_vectors(training_data, as_targets=False)

    assert torch.equal(
        lengths, torch.tensor([6, 5])
    )  # the length should include the start/stop token

    # check if the sentence is correctly encoded as input
    assert torch.equal(
        vectors,
        torch.tensor(
            [
                [1, 4, 5, 6, 7, 8],
                [1, 9, 5, 10, 11, 0],
            ]
        ),
    )


def test_target_vectors():
    training_data = [
        "this is a good movie".split(" "),
        "it is raining today".split(" "),
    ]

    vocabulary = Vocabulary.from_data(training_data)

    vectors, lengths = vocabulary.make_index_vectors(training_data, as_targets=True)

    assert torch.equal(
        lengths, torch.tensor([6, 5])
    )  # the length should include the start/stop token

    # check if the sentence is correctly encoded as input
    assert torch.equal(
        vectors,
        torch.tensor(
            [
                [4, 5, 6, 7, 8, 2],
                [9, 5, 10, 11, 2, 0],
            ]
        ),
    )


def test_split_sizes():
    corpus = list(range(16))
    a, b, c = split_corpus(corpus, (0.8, 0.1, 0.1))

    assert a == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert b == [13]
    assert c == [14, 15]

    assert a + b + c == corpus
