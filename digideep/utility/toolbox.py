import numpy as np
import torch, random
from digideep.utility.logging import logger
import inspect
import os
import json, yaml
import copy

def dump_dict_as_json(filename, dic, sort_keys=False):
    """
    This function dumps a python dictionary in ``json`` format to a file.

    Args:
        filename (path): The address to the file.
        dic (dict): The dictionary to be dumped in json format. It should be json-serializable.
        sort_keys(bool, False): Will sort the dictionary by its keys before dumping to the file.
    """
    f = open(filename, 'w')
    f.write(json.dumps(dic, indent=2, sort_keys=sort_keys))
    f.close()
def load_json_as_dict(filename):
    f = open(filename, 'r')
    try:
        dic = json.load(f)
    except json.JSONDecodeError as exc:
        print(exc)
        dic = {}
    f.close()
    return dic

def dump_dict_as_yaml(filename, dic):
    f = open(filename, 'w')
    f.write(yaml.dump(dic, indent=2))
    f.close()
def load_yaml_as_dict(filename):
    f = open(filename, 'r')
    try:
        # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        # dic = yaml.load(f, Loader=yaml.FullLoader)
        dic = yaml.load(f, Loader=yaml.UnsafeLoader)
    except yaml.YAMLError as exc:
        print(exc)
        dic = {}
    f.close()
    return dic

def seed_all(seed, cuda_deterministic = False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def get_module(addr):
    """
    Return a instance of a module by using only its name.
    
    Args:
        addr (str): The name of the module which should be in the format
          ``MODULENAME[.SUBMODULE1[.SUBMODULE2[...]]]``

    """
    parts = addr.split('.')
    module = ".".join(parts)
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_class(addr):
    """
    Return a instance of a class by using only its name.
    
    Args:
        addr (str): The name of the class/function which should be in the format
          ``MODULENAME[.SUBMODULE1[.SUBMODULE2[...]]].CLASSNAME``
    """
    parts = addr.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def count_parameters(model):
    """
    Counts the number of parameters in a PyTorch model.
    """
    return np.sum(p.numel() for p in list(model.parameters()) if p.requires_grad)


# def match_key(dict_target, dict_source, key, default):
#     if key in dict_source:
#         dict_target[key] = dict_source[key]
#         del dict_source[key]
#     else:
#         dict_target[key] = default

def strict_update(dict_target, dict_source):
    result = copy.deepcopy(dict_target)
    for key in dict_source:
        if key not in dict_target:
            logger.warn("The provided parameter '{}' was not available in the source dictionary.".format(key))
            # continue
        result[key] = dict_source[key]
    return result
        