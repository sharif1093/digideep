"""
This module provides helper functions to manage data outputs from the :class:`~digideep.environment.explorer.Explorer` class.

See Also:
    :ref:`ref-data-structure`

"""

import numpy as np
from collections import OrderedDict
from digideep.utility.logging import logger


############
## PART I ##
############
def join_keys(key1, key2, sep="/"):
    """
    Args:
        key1 (str): The first key in unix-style file system path.
        key1 (str): The second key in unix-style file system path.
        sep (str): The separator to be used.
    
    .. code-block:: python
        :caption: Example

        >>> join_keys('/agent1','artifcats')
        '/agent1/artifacts'
    
    """
    key1 = key1.rstrip(sep)
    key2 = key2.lstrip(sep)
    return key1+sep+key2

def flatten_dict(dic, sep="/", prefix=""):
    """
    We flatten a nested dictionary into a 1-level dictionary. In the new dictionary
    keys are combinations of previous keys, separated by the ``sep``. We follow unix-style
    file system naming.

    .. code-block:: python
        :caption: Example
        
        >>> Dict = {"a":1, "b":{"c":1, "d":{"e":2, "f":3}}}
        >>> flatten_dict(Dict)
        {"/a":1, "/b/c":1, "/b/d/e":2, "/b/d/f":3}
    
    """
    res = OrderedDict()
    for key, value in dic.items():
        if isinstance(value, dict):
            tmp = flatten_dict(value, sep=sep, prefix=join_keys(prefix,key,sep))
            res.update(tmp)
        else:
            res[join_keys(prefix,key,sep)] = value
    return res

def unflatten_dict(dic, sep="/"):
    """
    Unflattens a flattened dictionary into a nested dictionary.

    .. code-block:: python
        :caption: Example

        >>> Dict = {"/a":1, "/b/c":1, "/b/d/e":2, "/b/d/f":3}
        >>> unflatten_dict(Dict)
        {"a":1, "b":{"c":1, "d":{"e":2, "f":3}}}
    
    """
    def _insert(D, address, value):
        res = D
        parts = address.split(sep)
        parts = parts[1:] # Disregard the first as it should be empty
        for p in parts[:-1]: # Disregard the last which should contain the value finally.
            if not p in res:
                res[p] = {}
            res = res[p]
        res[parts[-1]] = value

    result = {}
    for key, value in dic.items():
        _insert(result, key, value)
    return result

#############
## PART II ##
#############
# The methods in this part are solely used in the Explorer class,
# where we want to form the chunks of data (trajectories). Memory
# has its own (faster) methods for update/complete the dicts.
def nonify(element):
    """
    This function creates an output with all elements being ``None``.
    The structure of the resulting element is exactly the structure
    of the input ``element``. The ``element`` cannot contain dicts.
    The only accepted types are ``tuple``, ``list``, and ``np.ndarray``.
    It can contain nested lists and tuples, however.

    .. code-block:: python
        :caption: Example

        >>> Input = [(1,2,3), (1,2,4,5,[-1,-2])]
        >>> nonify(Input)
        [(none,none,none), (none,none,none,none,[none,none])]
    
    """
    if isinstance(element, list) or isinstance(element, tuple):
        el = []
        for k in element:
            el.append(nonify(k))
        return el
    elif isinstance(element, np.ndarray) and np.issubdtype(element.dtype, np.floating):
        return np.full_like(element, fill_value=np.nan, dtype=element.dtype)
    elif isinstance(element, np.ndarray) and np.issubdtype(element.dtype, np.integer):
        return np.full_like(element, fill_value=0, dtype=element.dtype)
    elif isinstance(element, np.ndarray):
        return np.full_like(element, fill_value=np.nan, dtype=element.dtype)
    elif isinstance(element, float):
        return np.nan
    elif isinstance(element, int):
        return 0
    else:
        None

def update_dict_of_lists(dic, item, index=0):
    """
    This function updates a dictionary with a new item.

    .. code-block:: python
        :caption: Example

        >>> dic = {'a':[1,2,3], 'c':[[-1,-2],[-3,-4]]}
        >>> item = {'a':4, 'b':[1,2,3]}
        >>> index = 3
        >>> update_dict_of_lists(dic, item, index)
        {'a':[1,2,3,4],
         'b':[[none,none,none],[none,none,none],[none,none,none],[1,2,3]],
         'c':[[-1,-2],[-3,-4]]}
    
    Note:
        ``c`` in the above example is not "complete" yet! The function :func:`complete_dict_of_list`
        will complete the keys which need to be completed!
        
    Caution:
        This function does not support nested dictionaries.
    """
    for k in item:
        # 1. Create the "key" in the "dic" if it does not exist.
        #    Put "None" for all of the previous timesteps that
        #    the key was missing. Use the structure of the value
        #    to create a "None" element with the same structure.
        if not k in dic:
            none_element = nonify(item[k])
            dic[k] = [none_element] * index
        # 2. Now append the new value to the existing list of
        #    values for the key.
        dic[k].append(item[k])

def complete_dict_of_list(dic, length):
    """
    This function will complete the missing elements of a reference dictionary with similarly-structured ``None`` values.

    .. code-block:: python
        :caption: Example

        >>> dic = {'a':[1,2,3,4],
        ...        'b':[[none,none,none],[none,none,none],[none,none,none],[1,2,3]],
        ...        'c':[[-1,-2],[-3,-4]]}
        >>> # The length of lists under each key is 4 except 'c' which is 2. We have to complete that.
        >>> complete_dict_of_list(dic, 4)
        {'a':[1,2,3,4], 
         'b':[[none,none,none],[none,none,none],[none,none,none],[1,2,3]],
         'c':[[-1,-2],[-3,-4],[none,none],[none,none]]}
    """
    assert isinstance(dic, dict), "dic should be a dictionary."
    for k in dic:
        assert isinstance(dic[k], list), "dic[" + k + "] should be a list"
        if length > len(dic[k]):
            none_element = nonify(dic[k][-1])
            dic[k] += [none_element]*(length - len(dic[k]))


##############
## Part III ##
##############
def convert_time_to_batch_major(episode):
    """Converts a rollout to have the batch dimension in the major (first) dimension, instead of second dimension.

    Args:
        episode (dict): A trajectory in the form of ``{'key1':(num_steps,batch_size,...), 'key2':(num_steps,batch_size,...)}``
    
    Returns:
        dict: A trajectory in the form of ``{'key1':(batch_size,num_steps,...), 'key2':(batch_size,num_steps,...)}``
    
    .. code-block:: python
        :caption: Example

        >>> episode = {'key1':[[[1],[2]], [[3],[4]], [[5],[6]], [[7],[8]], [[9],[10]]],
                        'key2':[[[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]], [[13,14],[15,16]], [[17,18],[19,20]]]}
        >>> convert_time_to_batch_major(episode)
        {'key1': array([[[ 1.],
            [ 3.],
            [ 5.],
            [ 7.],
            [ 9.]],
    
            [[ 2.],
            [ 4.],
            [ 6.],
            [ 8.],
            [10.]]], dtype=float32), 'key2': array([[[ 1.,  2.],
            [ 5.,  6.],
            [ 9., 10.],
            [13., 14.],
            [17., 18.]],
    
            [[ 3.,  4.],
            [ 7.,  8.],
            [11., 12.],
            [15., 16.],
            [19., 20.]]], dtype=float32)}  
    """
    episode_batch = {}
    for key in episode.keys():
        try:
            # print(key, "=", episode[key])
            entry_data_type = episode[key][0].dtype

            val = np.array(episode[key], dtype=entry_data_type).copy()  #TODO: Should we copy?
            # make inputs batch-major instead of time-major
            episode_batch[key] = val.swapaxes(0, 1)
        except Exception as ex:
            logger.fatal('@', key, ':', ex)
            raise
        
    return episode_batch

def extract_keywise(dic, key):
    """This function will extract a key from all entries in a dictionary. Key should be first-level key.
    
    Args:
        dic (dict): The input dictionary containing a dict of dictionaries.
        key: The key name to be extracted.
    
    Returns:
        dict: The result dictionary

    .. code-block:: python
        :caption: Example

        >>> dic = {'agent1':{'a':[1,2],'b':{'c':2,'d':4}}, 'agent2':{'a':[3,4],'b':{'c':9,'d':7}}}
        >>> key = 'a'
        >>> extract_keywise(dic, key)
        {'agent1':[1,2], 'agent2':[3,4]}


    """
    res = {}
    for name in dic:
        res[name] = dic[name][key]
    return res

def dict_of_lists_to_list_of_dicts(dic, num):
    """Function to convert a dict of lists to a list of dicts.
    Mainly used to prepare actions to be fed into the ``env.step(action)``. ``env.step`` assumes
    action to be in the form of a list the same length as the number of workers. It will assign
    the first action to the first worker and so on.

    Args:
        dic (dict): A dictionary with keys being the actions for different agents in the environment.
        num (int): The number of workers.
    
    Returns:
        list: A length with its length being same as ``num``. Each element in the list would be a dictionary
        with keys being the agents.
    
    .. code-block:: python
        :caption: Example

        >>> dic = {'a1':([1,2],[3,4],[5,6]), 'a2':([9],[8],[7])}
        >>> num = 3
        >>> dict_of_lists_to_list_of_dicts(dic, num)
        [{'a1':[1,2], 'a2':[9]}, {'a1':[3,4], 'a2':[8]}, {'a1':[5,6], 'a2':[7]}]

    Caution:
        This only works for 1-level dicts, not for nested dictionaries.    
    """
    res = []
    for i in range(num):
        unit = {}
        for name in dic:
            unit[name] = dic[name][i]
        res.append(unit)
    return res

def list_of_dicts_to_flattened_dict_of_lists(List, length):
    """Function to convert a list of (nested) dicts to a flattened dict of lists. See the example below.

    Args:
        List (list): A list of dictionaries. Each element in the list is a single sample data produced from the environment.
        length (int): The length of time sequence. It is used to complete the data entries which were lacking from some data samples.
    Returns:
        dict: A dictionary whose keys are flattened similar to Unix-style file system naming.
    
    .. code-block:: python
        :caption: Example

        >>> List = [{'a':{'f':[1,2], 'g':[7,8]}, 'b':[-1,-2], 'info':[10,20]},
                    {'a':{'f':[3,4], 'g':[9,8]}, 'b':[-3,-4], 'step':[80,90]}]
        >>> Length = 2
        >>> list_of_dicts_to_flattened_dict_of_lists(List, length)
        {'/a/f':[[1,2],[3,4]],
        '/a/g':[[7,8],[9,8]],
            'b':[[-1,-2],[-3,-4]],
            '/info':[[10,20],[none,none]],
            '/step':[[none,none],[80,90]]}

    .. code-block:: python
        :caption: Example
        
        # Intermediate result, before doing ``complete_dict_of_list``:
        {'/a/f':[[1,2],[3,4]],
        '/a/g':[[7,8],[9,8]],
        'b':[[-1,-2],[-3,-4]],
        '/info':[[10,20]],
        '/step':[[none,none],[80,90]]}
        # Final result, after doing ``complete_dict_of_list`` ('/info' will become complete in length):
        {'/a/f':[[1,2],[3,4]],
        '/a/g':[[7,8],[9,8]],
        'b':[[-1,-2],[-3,-4]],
        '/info':[[10,20],[none,none]],
        '/step':[[none,none],[80,90]]}
    
    """
    # Does not support nested dictionaries (?)
    # This is used for info. But can be used for other list of dicts
    if isinstance(List, dict):
        return List
    Dict = OrderedDict()
    for i in range(len(List)):
        update_dict_of_lists(Dict, flatten_dict(List[i]), index=i)    
    # Here, complete_dict_of_list cannot be in the loop.
    # Since the new keys may arrive in a new list index.
    # For instance 'step' in List[1] in the above example.
    # And the "chunky" data will cause problems. Since
    # we may have a chunk of data without a key, but the
    # key arrives in a new chunk of data.
    # Here, we are outputing complete chunks of data.
    complete_dict_of_list(Dict, length=length)
    return Dict

def flattened_dict_of_lists_to_dict_of_numpy(dic):
    for key in dic:
        # dic[key] = np.asarray(dic[key], dtype=np.float32)
        dic[key] = np.asarray(dic[key])
    return dic