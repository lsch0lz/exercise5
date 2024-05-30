import random
from dataclasses import asdict, dataclass
from hashlib import shake_256
from typing import Optional, Tuple


@dataclass
class Configuration:
    """
    Stores the configuration for any given training run.

    Use `Configuration.generate` to pick unspecified parameters randomly.
    """

    seed: int = 0
    device: str = "cpu"
    dataset: str = "sanity_check"
    character_level: bool = False

    num_epochs: int = 50

    embedding_size: int = 64
    unk_threshold: int = 2
    minibatch_size: int = 1
    rnn_hidden_size: int = 50  # Hidden states in the LSTM
    learning_rate: float = 0.1
    num_layers: int = 1

    truncate_length: Optional[int] = 256  # Cut of tokens exceeding this length

    # Some meta information
    info: str = ""
    random_params: Tuple[str, ...] = ()

    @classmethod
    def username_to_seed(cls, username: str) -> int:
        return int.from_bytes(shake_256(username.encode("utf8")).digest(2), "big")

    @classmethod
    def generate(cls, username: str, iteration: Optional[int] = None, **kwargs):
        if iteration is not None:
            username = f"{username}/{iteration}"

        random.seed(cls.username_to_seed(username))

        random_params = ()

        RANDOM_INTERVALS = {
            "num_layers": (1, 4),
            "learning_rate": (-1.2, -0.2),
        }

        for name, interval in RANDOM_INTERVALS.items():
            if name not in kwargs:
                if name == "learning_rate":
                    exp = random.random() * (interval[1] - interval[0]) + interval[0]
                    kwargs[name] = 10**exp
                else:
                    kwargs[name] = random.randint(*interval)

                random_params += (name,)

        if "rnn_hidden_size" not in kwargs:
            kwargs["rnn_hidden_size"] = 600 // kwargs["num_layers"]

        return cls(**kwargs, info=username, random_params=random_params)

    def __str__(self):
        lines = ["Configuration:"] + [f"- {k}: {v}" for k, v in asdict(self).items()]

        return "\n".join(lines)
