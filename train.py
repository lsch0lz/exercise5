import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import trange

from config import Configuration
from data_util import Sentence, Vocabulary, load_corpus, split_corpus
from model import LanguageModel
from result_util import produce_result


def prepare_batch(
    *, sentences, vocabulary, config
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    input_ids, input_lengths = vocabulary.make_index_vectors(
        sentences, as_targets=False
    )
    target_ids, _ = vocabulary.make_index_vectors(sentences, as_targets=True)

    input_ids = input_ids.to(config.device)
    target_ids = target_ids.to(config.device)

    return input_ids, input_lengths, target_ids


def train(
    *,
    model: torch.nn.Module,
    vocabulary: Vocabulary,
    config: Configuration,
    training_data: List[Sentence],
    validation_data: List[Sentence],
    model_checkpoint_path: str = "best-model.pt",
):
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)

    # Log the losses and accuracies
    stats = {
        "train_loss": [],
        "validation_loss": [],
        "validation_perplexity": [],
        "epoch_duration": [],
        "best_epoch": -1,
        "best_validation_perplexity": 100000.0,
    }

    # Go over the training dataset multiple times
    for epoch in range(config.num_epochs):
        print(f"--- Epoch {epoch} ---")

        model.train()

        start = time.time()

        # Shuffle training data at each epoch
        random.shuffle(training_data)

        train_loss = 0.0

        # Each forward pass is now over a batch of sentences
        for i in trange(0, len(training_data), config.minibatch_size):
            sentences = training_data[i : i + config.minibatch_size]

            # Remember that PyTorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Make the input and target vectors
            input_ids, input_lengths, target_ids = prepare_batch(
                sentences=sentences, vocabulary=vocabulary, config=config
            )

            # Calculate loss
            output = model.forward(input_ids, input_lengths, target_ids=target_ids)
            loss = output["loss"]
            train_loss += loss.item()

            # Compute the backward pass and adjust the parameters
            loss.backward()
            optimizer.step()

        train_loss /= len(training_data)

        # Evaluate and print accuracy at end of each epoch
        val_ppl, val_loss = evaluate(
            model=model, vocabulary=vocabulary, data=validation_data, config=config
        )

        # If the current model has the lowest valudation perplexity, save it
        if val_ppl < stats["best_validation_perplexity"]:
            print("new best model found!")
            stats["best_epoch"] = epoch
            stats["best_validation_perplexity"] = val_ppl
            model.save_checkpoint(model_checkpoint_path)

        duration = time.time() - start

        print(f"training loss: {train_loss}")
        print(f"validation loss: {val_loss}")
        print(f"validation perplexity: {val_ppl}")
        print(f"{round(duration, 3)} seconds for this epoch")

        # Append values to lists for later plots
        stats["train_loss"].append(train_loss)
        stats["validation_loss"].append(val_loss)
        stats["validation_perplexity"].append(val_ppl)
        stats["epoch_duration"].append(duration)

    # Load the best model
    model.load_checkpoint(model_checkpoint_path)

    return stats


def evaluate(*, model, vocabulary, data, config) -> Tuple[float, float]:
    """Evaluate the model on some data (and return the perplexity and loss)."""

    model.eval()
    with torch.no_grad():  # Do not store activations to compute the backward pass
        aggregate_loss = 0.0

        # Go through all test data points
        for sentence in data:
            # Make input and target vectors
            input_ids, input_lengths, target_ids = prepare_batch(
                sentences=[sentence], vocabulary=vocabulary, config=config
            )

            # Calculate loss
            output = model.forward(input_ids, input_lengths, target_ids=target_ids)
            aggregate_loss += output["loss"].item()

        # Compute the average loss over all data points
        aggregate_loss = aggregate_loss / len(data)

        return math.exp(aggregate_loss), aggregate_loss


def do_run(
    config: Configuration,
    results_directory="runs",
    show_plot: bool = True,
):
    torch.manual_seed(config.seed)

    # Data filename
    data_path = Path("data") / f"{config.dataset}.txt"

    # Load the corpus and split it into a training, validation, and test portion
    complete_corpus = load_corpus(
        data_path,
        character_level=config.character_level,
        truncate_length=config.truncate_length,
    )
    training_data, validation_data, test_data = split_corpus(complete_corpus)

    print(
        f"Training corpus has {len(training_data)} train, {len(validation_data)} validation and {len(test_data)} test sentences.\n"
    )

    print(f"Longest sentence has {max(len(s) for s in training_data)} tokens.")

    # Create the vocabulary (token dictionary) from the training data
    vocabulary = Vocabulary.from_data(training_data, unk_threshold=config.unk_threshold)

    print(f"Using a vocabulary with {len(vocabulary)} tokens.")

    # Create the language model
    model = LanguageModel(
        vocab_size=len(vocabulary),
        pad_index=vocabulary[vocabulary.PAD_TOKEN],
        embedding_size=config.embedding_size,
        rnn_hidden_size=config.rnn_hidden_size,
        is_character_level=config.character_level,
        num_layers=config.num_layers,
    )
    model.to(config.device)  # Move it to the target device

    # Print the configuration and start the training loop
    print(str(config), "\n")
    print("=== Starting Training ===\n")

    stats = train(
        model=model,
        vocabulary=vocabulary,
        config=config,
        training_data=training_data,
        validation_data=validation_data,
    )

    # Evaluate the final model
    stats["test_perplexity"], stats["test_loss"] = evaluate(
        model=model, vocabulary=vocabulary, data=test_data, config=config
    )

    print("\nTraining Complete.\n\n=== Evaluating ===")
    print(f" - using model from epoch {stats['best_epoch']} for final evaluation")
    print(f" - final score: {stats['test_perplexity']}")

    # Write the configuration to file & plot the loss and perplexity curves
    produce_result(
        config=config,
        stats=stats,
        results_directory=results_directory,
        show_plot=show_plot,
    )

    # If dataset is sanity_check, it should generate "this is not a love song"
    if config.dataset == "sanity_check":
        model.generate_text(vocabulary=vocabulary, prefix="this is")

    # If dataset is equations, it should generate actual match
    if config.dataset == "equations":
        model.generate_text(vocabulary=vocabulary, prefix="one plus one equals")
        model.generate_text(vocabulary=vocabulary, prefix="two plus two equals")
        model.generate_text(vocabulary=vocabulary, prefix="three plus three equals")
        model.generate_text(vocabulary=vocabulary, prefix="four plus four equals")
        model.generate_text(vocabulary=vocabulary, prefix="five plus five equals")

    # If dataset is motivational_quotes, it should generate ... motivation quotes?
    if config.dataset == "motivational_quotes":
        model.generate_text(vocabulary=vocabulary, prefix="life is")
        model.generate_text(vocabulary=vocabulary, prefix="marriage is")
        model.generate_text(vocabulary=vocabulary, prefix="luck is")

    return model, vocabulary


if __name__ == "__main__":
    # TODO: Set your github username
    username = "lsch0lz"

    config = Configuration.generate(
        username=username,
        num_epochs=50,
        minibatch_size=32,
        dataset="sanity_check",
        character_level=False,
        device="cpu",
    )

    do_run(config)
