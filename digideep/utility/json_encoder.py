import json
import numpy as np

class MultiDimensionalArrayEncoder(json.JSONEncoder):
    """
    A ``JSONEncoder`` which ca serialize tuples. See `Stack Overflow`_ post for more information.

    .. _Stack Overflow: https://stackoverflow.com/a/15721641
    """
    def encode(self, obj):
        def hint_tuples(item):
            # print("item =", item, ",", type(item))
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            elif isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            elif isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            elif isinstance(item, np.ndarray):
                return {'__numpy__': True, 'items': item.tolist()}
            elif type(item).__module__ == np.__name__:
                return {'__numpy__': True, 'items': item.item()}
            else:
                return item
        return super(MultiDimensionalArrayEncoder, self).encode(hint_tuples(obj))

def hinted_tuple_hook(obj):
    """
    Args:
        obj: ``'{"a":12, "beta":{"__tuple__": True, "items": [0.9, 0.99]}}'``
    """
    if '__tuple__' in obj:
        return tuple(obj['items'])
    elif '__numpy__' in obj:
        return np.array(obj['items'])
    else:
        return obj


enc = MultiDimensionalArrayEncoder()
JsonEncoder = lambda obj: enc.encode(obj)
JsonDecoder = lambda jsonstring: json.loads(jsonstring, object_hook=hinted_tuple_hook)
