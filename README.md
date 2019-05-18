> **Caution**: Code is under active development. Breaking changes are probable.

[![Documentation Status](https://readthedocs.org/projects/digideep/badge/?version=latest)](https://digideep.readthedocs.io/en/latest/?badge=latest)

# Digideep

A pipeline for fast prototyping Deep RL problems using [PyTorch](https://github.com/pytorch/pytorch)
and [OpenAI's Gym](https://github.com/openai/gym) / [Deepmind's dm_control](https://github.com/deepmind/dm_control).

Digideep is written to be developer-friendly with self-descriptive codes and extensive documentation. It also provides
some debugging tools and guidelines for implementing new methods. Digideep has the following methods implemented:

* DDPG - Deep Deterministic Policy Gradient
* PPO - Proximal Policy Optimization

Please use the following BibTeX entry to cite this repository in your publications:

```bibtex
@misc{digideep19,
  author = {Sharif, Mohammadreza},
  title = {Digideep: A pipeline for implementing reinforcement learning problems},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/sharif1093/digideep}},
}
```

## Documentation

Please visit https://digideep.readthedocs.io/en/latest/ for documentation.

## Features

The features of Digideep can be listed as following:

* Developer-friendly code:
  * The code is highly readable and fairly easy to understand and modify.
  * Extensive documentation to support the above.
  * Written for modularity and easy code reuse.
  * Provides debugging tools as an assistance for implementation new methods.
* Has a single-node multi-cpu multi-gpu architecture implemented to utilize CPU and GPU on a single node.
* Connects to `dm_control` and uses `dm_control`'s native viewer for rendering.
* Provides batch-environment for `dm_control` through OpenAI baseline's `VecEnv` wrapper.
* Controls all parameters from one single `parameter` file for easier control.
* Supports (de-)serialization structurally.


## Changelog

* **_2019-03-04_**: Digideep was launched.

## License

This code is under BSD 2-clause (FreeBSD/Simplified) license. See [LICENSE](LICENSE).

## Acknowledgement

I would like to appreciate authors of
[OpenAI baselines](https://github.com/openai/baselines), 
[pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr),
[RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2), and
[RLkit](https://github.com/vitchyr/rlkit) projects.
