[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/frklxFXf)
# Exercise 5

This exercise contains three tasks: (1) Implement the data preprocessing and the forward pass of the language model, (2) train an LM on a small dataset and compare character vs word-level tokens, and (3) train an LM on a larger dataset comparing different batch sizes.

## Instructions

Accept the assignment in GitHub classrooms (link on exercise sheet and slides). Clone the repository to your local machine. All code completion tasks are required with a TODO-tag inside the code.

For task 1, commit your code to the repository. For task 2 and 3, commit the directory `runs` with the 4 runs that should be completed.

Please consult the exercise sheet for further details.

## Hyperparameters

For task 2 and 3, we expect you to provide 2 runs each -- keep all parameters which do not need to be modified at the default value.

You can run the training using the following code snippets:

```python
from config import Configuration
from train import do_run

username = "someuser"  # TODO: specify your github username here so every students runs are unique

do_run(Configuration.generate(
    username=username, dataset="equations", character_level=False,
))

do_run(Configuration.generate(
    username=username, dataset="equations", character_level=True,
))

do_run(Configuration.generate(
    username=username, dataset="motivational_quotes", minibatch_size=4,
))

do_run(Configuration.generate(
    username=username, dataset="motivational_quotes", minibatch_size=32,
))

```
