> **Caution 1**: Code is under active development. Breaking changes are probable.
>
> **Caution 2**: Documentation is lagging behind the code development.

[![Documentation Status](https://readthedocs.org/projects/digideep/badge/?version=latest)](https://digideep.readthedocs.io/en/latest/?badge=latest)

# Digideep

## Introduction

Developers who want to implement a new deep DeepRL algorithm, usually have to write a great amount of boilerplate code or alternatively use 3rd party packages which aim to provide the basics. However, understanding and modifying these 3rd party packages usually is not a trivial task, due to lack of either documentation or code readability/structure.

Digideep tries to provide a well-documented complete pipeline for deep reinforcement learning problems, so that developers can jump directly to implementing their methods. Special attention has been paid to **decoupling** different components as well as making them **modular**.

In Digideep, [OpenAI's Gym](https://github.com/openai/gym) and [Deepmind's dm_control](https://github.com/deepmind/dm_control) co-exist and can be used with the same interface. Thanks to decoupled simulation and training parts, both [TensorFlow ](https://www.tensorflow.org/) and [PyTorch](https://github.com/pytorch/pytorch) can be used to train the agents (however, the example methods in this code repository are implemented using PyTorch).

Currently, the following methods are implemented in Digideep:

* [DDPG](https://arxiv.org/abs/1509.02971) - Deep Deterministic Policy Gradient
* [SAC](https://arxiv.org/abs/1801.01290) - Soft Actor Critic
* [PPO](https://arxiv.org/abs/1707.06347) - Proximal Policy Optimization

Digideep is written to be developer-friendly with self-descriptive codes and extensive documentation. It also provides
some debugging tools and guidelines for implementing new methods.

## Features

* Developer-friendly code:
  * The code is highly readable and fairly easy to understand and modify.
  * Extensive documentation to support the above.
  * Written for _modularity_ and code _decoupling_.
  * Provides _debugging tools_ as an assistance for implementation new methods.
* Supports single-node multi-cpu multi-gpu architecture.
* Supports _dictionary observation/action spaces_ for neat communication with environments.
* Can be used with both `dm_control`/`gym` using the same interface:
  * Uses `dm_control`'s native viewer for viewing.
  * Provides batch environments for both `dm_control` and `gym`.
* Provides a session-as-a-module (SaaM) functionality to easily load saved sessions as a Python module for post-processing.
* Controls all parameters from a _single `parameter` file_ for transparency and easy control of all parameters from one place.
* Supports _(de-)serialization_ structurally.

## Documentation

Please visit https://digideep.readthedocs.io/en/latest/ for documentation.

## Changelog

* **_2019-05-20_**: Added Soft Actor-Critic (SAC). Added full support for Dict observation spaces.
* **_2019-03-04_**: Digideep was launched.

## Contributions

Contributions are welcome. If you would like to contribute to Digideep consider [Pull Requests](https://github.com/sharif1093/digideep/pulls) and [Issues](https://github.com/sharif1093/digideep/issues) pages of the project.

## Citation

Please use the following BibTeX entry to cite this repository in your publications:

```bibtex
@misc{digideep19,
  author = {Sharif, Mohammadreza},
  title = {Digideep: A DeepRL pipeline for developers},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sharif1093/digideep}},
}
```

## License

BSD 2-clause.

## Acknowledgement

I would like to appreciate authors of
[OpenAI baselines](https://github.com/openai/baselines), 
[pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2), and
[RLkit](https://github.com/vitchyr/rlkit) projects.
