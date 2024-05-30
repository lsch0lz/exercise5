import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt


def get_free_path(base_path) -> Path:
    """Get an unoccupied path in the results directory to save training artefacts."""
    for i in range(10000):
        path = Path(base_path) / f"{i:04}"

        if path.exists():
            continue

        path.mkdir(parents=True)
        return path

    msg = "You should take a break!"
    raise RuntimeError(msg)


def produce_result(*, config, stats, results_directory="runs", show_plot: bool = True):
    path = get_free_path(results_directory)

    print("Results stored at:", path)

    # Store the results
    results = {
        "config": asdict(config),
        "stats": stats,
    }
    with open(path / "result.json", "w") as json_file:
        json.dump(results, json_file)

    # Write a readable form of the configuration
    with open(path / "config.txt", "w") as text_file:
        print(str(config), file=text_file)

    # Create performance plots
    plot_loss_curves(config=config, stats=stats)

    plt.savefig(path / "curves.png")

    if show_plot:
        plt.show()


def plot_loss_curves(*, config, stats):
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax1.plot(stats["train_loss"], "r", label="Training loss")
    ax1.plot(stats["validation_loss"], "g", label="Validation loss")

    ax1.grid()

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Perplexity")  # We already handled the x-label with ax1
    ax2.plot(stats["validation_perplexity"], "black", label="Validation Perplexity")
    ax2.tick_params(axis="y")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)

    fig.tight_layout()
