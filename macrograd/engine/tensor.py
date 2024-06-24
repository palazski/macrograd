from typing import Union


class Tensor:

    # TODO: Maybe pass shape instead of value?
    def __init__(self, data=None) -> None:
        self.data = data

        self.shape = None
        if data:
            self._set_shape()

        self.grad = None
        self.backward = None

    def _set_shape(self, shape=None):
        if shape:
            assert isinstance(shape, tuple, list, int), 'shape must be of type tuple, list, or int'
            self.shape = shape
            return shape

        # TODO: Make this better
        def get_shape(lst):
            if not isinstance(lst, list):
                return (1)
            if not lst:
                return ()
            shape = []
            sub_list = lst
            while isinstance(sub_list, list):
                shape.append(len(sub_list))
                if not sub_list:  # If any dimension has 0 elements, stop here
                    break
                sub_list = sub_list[0]

            shape = tuple(shape)
            return shape
        
        self.shape = get_shape(self.data)

    def _set_value(self, value):
        assert isinstance(value, (list, int, float)), 'tensor input cannot be of not number'
        if not isinstance(value, list):
            value = [value]
        
        self.data = value
        self._set_shape()

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) > len(self.shape):
            raise ValueError(f'cannot slice ({len(key)}) more than shape of the Tensor ({len(self.shape)})')
        
        if isinstance(key, (slice, int)) and len(self.shape) < 1:
            raise ValueError(f'cannot slice (1) more than shape of the Tensor (< 1)')
        
        def _handle_slice_index(key: slice, len_data: int) -> slice:
            step = 1 if not key.step else key.step

            start = key.start
            if not start:
                start = 0 if step > 0 else len_data - 1

            stop = key.stop
            if not stop:
                stop = len_data + 1 if step > 0 else -len_data - 1

            if start < 0:
                while start > 0:
                    start = len_data + start

            if stop < 0:
                while stop > 0:
                    stop = len_data + stop

            return slice(start, stop, step)
        
        def __handle_integer_index(key: int, len_data: int) -> int:
            if key > len_data:
                # raise IndexError(f'index 3 is out of bounds for axis 0 with size 3')  # TODO: Give axis information
                raise IndexError(f'index {key} is out of bounds for axis with size {len_data}')
            
            if key < 0:
                key = key + len_data
            
            return key

        val_return = None
        if isinstance(key, slice):
            key = _handle_slice_index(key, len(self.data))
            val_return = self.data[key.start:key.stop:key.step]
        elif isinstance(key, tuple):
            val_return = self.data
            low_rank = False
            for dim in key:
                print(f'Current val: {val_return}')
                print(f'low rank: {low_rank}')
                if isinstance(dim, slice):
                    # listeyken slice ile icinde liste donduysen rank dustu
                    # listeyken slice ile liste donmediysen rank sabit

                    # su an rank dusmuyor.........
                    if low_rank:
                        while len(val_return) == 1:
                            val_return = val_return[0]
                    dim = _handle_slice_index(dim, len(val_return))
                    print(f"Inner slice is: {dim}")
                    val_return = [item[dim.start:dim.stop:dim.step] for item in val_return] if low_rank else val_return[dim.start:dim.stop:dim.step]
                    low_rank = isinstance(val_return[0], list)

                elif isinstance(dim, int):
                    print(f"Inner int dim is: {dim}")
                    dim = __handle_integer_index(dim, len(val_return))
                    val_return = [item[dim] for item in val_return] if low_rank else val_return[dim]
                else:
                    raise ValueError(f'slicing with {type(dim)} is not supported')
        elif isinstance(key, int):
            key = __handle_integer_index(key, len(self.data))
            val_return = self.data[key]
        else:
            raise ValueError(f'slicing with {type(key)} is not supported')
        
        if val_return is None:
            raise ValueError(f'return value of none!')
        
        return Tensor(val_return)

    def __len__(self):
        return len(self.data)

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
        
        value = zero_fill(shape)
        tensor = Tensor()
        tensor._set_value(value)

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
