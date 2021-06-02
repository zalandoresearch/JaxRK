from typing import Any, Tuple, Callable, Union

__all__ = ["Bijection", "PRNGKeyT", "Shape", "Dtype", "Array", "InitFn", "ConstOrInitFn", "FloatOrInitFn", "AnyOrInitFn"]


PRNGKeyT = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

InitFn = Callable[..., Any]

ConstOrInitFn = Union[float, InitFn, Any]
FloatOrInitFn = Union[float, InitFn]
AnyOrInitFn = Union[Any, InitFn]


class Bijection(object):
    def __call__(self, x):
        raise NotImplementedError
        
    def inv(self, y):
        raise NotImplementedError