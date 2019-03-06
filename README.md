**Caution**: Code is under active development. Breaking changes are probable.

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
  * The code is highly readable and easy to learn and modify.
  * Extensive documentation and tutorials for getting developers engaged in coding faster.
  * Different parts are decoupled and modular, so that developers can focus only on method
    implementations rather than exploration/storage.
  * Provides debugging tools for profiling time and monitoring values.
* Has single-node multi-cpu multi-gpu architectures implementation to utilize CPU and GPU better on a single node.
* Connects to `dm_control` and uses `dm_control`'s native viewer for rendering.
* It provides a multi-agent architecture.
* Controling all parameters from withing one single `parameter` file.
* Structured periodic saving of results as sessions.
* Serialization is a built-in feature of Digideep.


## Changelog

* **_2019-03-04_**: Digideep was launched.

## License

This code is under BSD 2-clause (FreeBSD/Simplified) license. See [LICENSE](LICENSE).

## Acknowledgement

I would like to appreciate authors of [OpenAI baselines](https://github.com/openai/baselines) and 
[pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr) projects, by which
this project is highly inspired.
