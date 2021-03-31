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
from jaxrk.utilities.views import tile_view
from jaxrk.core import Module
import flax.linen as ln
from jaxrk.core.typing import PRNGKeyT, Shape, Dtype, Array, ConstOrInitFn
from jaxrk.core.init_fn import ConstFn





class Reduce(Callable, ABC, Module):
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

# class SeqReduce(Reduce):
#     children:List[Reduce]
    
#     def __call__(self, inp:np.array, axis:int = 0) -> np.array:
#         carry = np.swapaxes(inp, axis, 0)
#         if self.children is not None:
#             for gr in self.children:
#                 carry = gr.reduce_first_ax(carry)
#         return np.swapaxes(carry, axis, 0)
    
#     def new_len(self, original_len:int):
#         carry = original_len
#         if self.children is not None:
#             for gr in self.children:
#                 carry = gr.new_len(carry)
#         return carry

class NoReduce(Reduce):
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return inp

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return inp
    
    def new_len(self, original_len:int) -> int:
        return original_len

class Prefactors(Reduce):
    factors_init: ConstOrInitFn = ConstFn(np.ones(1))
    dim:int = None

    def setup(self):
        if not isinstance(self.factors_init, Callable):
            if self.dim is None:
                dim = self.factors_init.shape[0]
            else:                
                dim = self.dim
        self.prefactors = self.const_or_param("prefactors", self.factors_init, (self.dim,), np.float32, )
        assert dim == self.prefactors.shape[0]

    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        assert self.prefactors.shape[0] == inp.shape[axis]
        return inp.astype(self.prefactors.dtype) * np.expand_dims(self.prefactors, axis=(axis+1)%2)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.__call__(inp, 0)
    
    def new_len(self, original_len:int) -> int:
        assert original_len == len(self.prefactors)
        return original_len

class Repeat(Reduce):
    times:int

    def setup(self, ):
        assert self.times > 0
    
    def __call__(self, inp:np.array, axis:int = 0) -> np.array:
        return np.repeat(inp, axis)

    def reduce_first_ax(self, inp:np.array) -> np.array:
        return self.call(inp, 0)
    
    def new_len(self, original_len:int) -> int:
        return original_len * self.times

class TileView(Reduce):
    new_len:int

    def setup(self, ):
        assert self.new_len > 0

    def reduce_first_ax(self, inp:np.array) -> np.array:
        assert self.new_len % inp.shape[0] == 0, "Input can't be broadcasted to target length %d" % self._len
        return tile_view(inp, self.new_len//inp.shape[0])
    
    def new_len(self, original_len:int) -> int:
        return self.new_len


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
    points_per_split:int
    
    def setup(self, ):
        assert self.points_per_split > 0

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