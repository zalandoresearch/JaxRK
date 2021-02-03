"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable, List
from abc import ABC, abstractmethod

import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp





class Reduce(Callable, ABC):
    """The basic reduction type."""
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        rval = self.reduce_first_ax(np.swapaxes(inp, axis, 0))
        return np.swapaxes(rval, axis, 0)

    @abstractmethod
    def reduce_first_ax(self, gram:np.array) -> np.array:
        pass

    @abstractmethod
    def new_len(self, original_len:int) -> int:
        pass

    @classmethod
    def apply(cls, inp, reduce:List["Reduce"] = None, axis = 0):
        carry = np.swapaxes(inp, axis, 0)
        if reduce is not None:
            for gr in reduce:
                carry = gr.reduce_first_ax(carry)
        return np.swapaxes(carry, axis, 0)
    
    @classmethod
    def final_len(cls, original_len:int, reduce:List["Reduce"] = None):
        carry = original_len
        if reduce is not None:
            for gr in reduce:
                carry = gr.new_len(carry)
        return carry

class NoReduce(Reduce):
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return inp

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return inp
    
    def new_len(self, original_len:int) -> int:
        return original_len

class Prefactors(Reduce):
    def __init__(self, pref:np.array) -> None:
        assert len(pref.shape) == 1
        self.prefactors = pref

    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return inp.astype(self.prefactors.dtype) * np.expand_dims(self.prefactors, axis=(axis+1)%2)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.__call__(inp, 0)
    
    def new_len(self, original_len:int) -> int:
        assert original_len == len(self.prefactors)
        return original_len

class Repeat(Reduce):
    def __init__(self, times):
        super().__init__()
        self.times = times
    
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return np.repeat(inp, axis)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.call(inp, 0)
    
    def new_len(self, original_len:int) -> int:
        return original_len * self.times

class TileView(Reduce):
    def __init__(self, new_len:int):
        super().__init__()
        self._len = new_len

    def reduce_first_ax(self, inp:np.array) -> np.array:
        assert self._len % inp.shape[0] == 0, "Input can't be broadcasted to target length %d" % self._len
        return np.broadcast_to(inp.ravel(), (self._len//inp.shape[0], inp.size)).reshape((self._len, inp.shape[1]))
    
    def new_len(self, original_len:int) -> int:
        return self._len


class Sum(Reduce):
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return inp.sum(axis = axis, keepdims = True)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.__call__(inp, 0)
    
    def new_len(self, original_len:int) -> int:
        return 1

class Mean(Reduce):
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return inp.mean(axis = axis, keepdims = True)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.__call__(inp, 0)
    
    def new_len(self, original_len:int) -> int:
        return 1

class BalancedSum(Reduce):
    """Sum up even number of elements in input."""
    def __init__(self, points_per_split:int) -> None:
        assert points_per_split > 0
        self.points_per_split = points_per_split

    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        perm = list(range(len(inp.shape)))
        perm[0] = axis
        perm[axis] = 0


        inp = np.transpose(inp, perm)
        inp = np.sum(np.reshape(inp, (-1, self.points_per_split, inp.shape[-1])), axis = 1) 
        inp =  np.transpose(inp, perm)
        return inp

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.__call__(inp, 0)

    def new_len(self, original_len:int) -> int:
        assert original_len % self.points_per_split == 0
        return original_len // self.points_per_split

class Center(Reduce):
    """Center input along axis."""
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return inp - np.mean(inp, axis, keepdims = True)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.__call__(inp, 0)

    def new_len(self, original_len:int) -> int:
        return original_len