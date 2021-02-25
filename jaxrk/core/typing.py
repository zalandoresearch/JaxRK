from typing import Any, Tuple, Callable, Union

__all__ = ["Bijection", "PRNGKeyT", "Shape", "Dtype", "Array", "InitFn", "ConstOrInitFn", "FloatOrInitFn", "ArrayOrInitFn"]


PRNGKeyT = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

InitFn = Callable[..., Array]

ConstOrInitFn = Union[float, Array, InitFn]
FloatOrInitFn = Union[float, InitFn]
ArrayOrInitFn = Union[float, InitFn]


class Bijection(object):
    def __call__(self, x):
        raise NotImplementedError
        
    def inv(self, y):
        raise NotImplementedError