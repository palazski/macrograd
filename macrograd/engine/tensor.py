from typing import Union


class Tensor:

    # TODO: Maybe pass shape instead of value?
    def __init__(self, data=None) -> None:
        self._set_value(data)

        self.grad = None
        self.backward = None

    def _set_shape(self, shape=None):
        if shape:
            assert isinstance(shape, (tuple, list, int)), 'shape must be of type tuple, list, or int'
            self.shape = shape
            return shape
        
        shape = []
        d = self.data
        while isinstance(d, list):
            shape.append(len(d))
            if d:
                d = d[0]
            else:
                break

        self.shape = tuple(shape)

    def _set_value(self, value):
        assert isinstance(value, (list, int, float)), 'tensor input cannot be of not number'
        if not isinstance(value, list):
            value = [value]

        # TODO: Check whether every dimension of the data is fine, we cannot do something like [[1, 2], [3, 4, 5]]
        self.data = value
        self._set_shape()

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        def recursive_slice(data, indices):
            if not isinstance(data, list):
                return data

            if not indices:
                return data

            current_index = indices[0]
            if isinstance(current_index, slice):
                return [recursive_slice(item, indices[1:]) for item in data[current_index]]
            elif isinstance(current_index, int):
                return recursive_slice(data[current_index], indices[1:])
            else:
                raise TypeError("Invalid index type")

        result_data = recursive_slice(self.data, key)
          
        return Tensor(result_data)

    def __len__(self):
        return self.shape[0]

    def __add__(self, other):
        assert isinstance(other, Tensor), 'summation is being support only with another Tensor'

    def __repr__(self):
        return str(self.data)
    
    def __str__(self):
        return str(self.data)
    
    def tolist(self):
        return self.data

    @classmethod
    def from_zeros(shape):
        def zero_fill(shape):
            if len(shape) == 1:
                return [0] * shape[0]
            else:
                return [zero_fill(shape[1:]) for _ in range(shape[0])]
        
        tensor = Tensor(zero_fill(shape))
        return tensor



def _broadcast(t: Tensor, shape: tuple):
    assert t.shape, f'tensor {t} is not associated with a shape'

    # broadcast_shape = _get_broadcast_shape(t, shape)  # checks for errors as well

    # assume can be broadcasted
    for i, dim in enumerate(shape):
        d = t
        print(f'type d: {type(d)}')
        for x in range(i):
            d = d[:, 0]
            print(f'type d: {type(d)}')
        print(f'd in depth {i}: {d}')
        print(f't.shape[i] is {t.shape[i]}')
        if t.shape[i] == 1:
            d = [[d[j] for i in range(dim)] for j in range(len(d))]
            t._set_value(d)
        print(f'new d: {d}')
        print(f't: {t}')
        print('-'*20)


def _get_broadcast_shape(t: Union[Tensor, tuple], z: Union[Tensor, tuple]):
    if isinstance(t, Tensor):
        assert t.shape, f'tensor {t} must have defined shape value'
        t = t.shape
    
    if isinstance(z, Tensor):
        assert z.shape, f'tensor {z} must have defined shape value'
        z = z.shape
    
    shape_s = z if len(t) > len(z) else t
    shape_l = z if len(t) <= len(z) else t

    shape_s = (1,) * (len(shape_l) - len(shape_s)) + shape_s  # Pad small shape with ones
    
    if shape_s == shape_l:
        return shape_l

    # broadcast_shape = ()
    # for dim_s, dim_l in zip(shape_s[::-1], shape_l[::-1]):
    #     if dim_s == dim_l or dim_l == 1 or dim_s == 1:
    #         broadcast_shape = (max(dim_s, dim_l),) + broadcast_shape
    #     else:
    #         raise ValueError(f'shapes cannot be broadcasted: {t.shape} {z.shape}')
    
    # This is a ~9x faster implementation, should be tested
    broadcast_shape = [max(dim_s, dim_l) for dim_s, dim_l in zip(shape_s[::-1], shape_l[::-1]) if dim_s == dim_l or dim_l == 1 or dim_s == 1]
    if len(shape_l) > len(broadcast_shape):
        raise ValueError(f'shapes cannot be broadcasted: {t} {z}')
    broadcast_shape = tuple(broadcast_shape)
    return broadcast_shape
