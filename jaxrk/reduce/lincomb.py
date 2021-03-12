"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable, List, TypeVar, Tuple

from jax import jit
from functools import partial
import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from jax.lax import scan, map
from jax import vmap
from jax.ops import index_update
from jaxrk.core.typing import  Array


from .base import Reduce

ListOfArray_or_Array_T = TypeVar("CombT", List[Array], Array)

class SparseReduce(Reduce):
    """SparseReduce constructs a Gram matrix by summing/averaging over rows of its input

        Args:
            idcs (List[np.array]): The indices of the rows to sum/average in the desired order. Each list element contains 2d arrays. The number of columns in the array is the number of summed/averaged elements.
            average (bool): If True average rows, else sum rows."""

    idcs:List[Array]
    average:bool = True
    
    def setup(self):
        self.idcs = idcs
        max_idx_const = self.variable('constants', "max_idx_const", lambda idcs: np.max(np.array([np.max(i) for i in idcs])), self.idcs)
        self.max_idx = max_idx_const.value
        if self.average:
            self._reduce = np.mean
        else:
            self._reduce = np.sum

    def reduce_first_ax(self, inp:np.array) -> np.array:
        assert (self.max_idx + 1) <= len(inp), self.__class__.__name__ + " expects a longer gram to operate on"
        assert len(inp.shape) == 2
        rval = []

        for i in range(len(self.idcs)):
            if self.idcs[i].shape[1] == 0:
                rval.append(np.zeros((self.idcs[i].shape[0], inp.shape[1])))
            else:
                reduced = self._reduce(inp[list(self.idcs[i].flatten()),:].reshape((-1, self.idcs[i].shape[1], inp.shape[1])), 1)
                rval.append(reduced)
        return np.concatenate(rval, 0)
    
    def new_len(self, original_len:int):
        assert (self.max_idx + 1) <= original_len, self.__class__.__name__ + " expects a longer gram to operate on"
        return len(self.idcs)
    
    
    @classmethod
    def sum_from_unique(cls, input:np.array, mean:bool = True) -> Tuple[np.array, np.array, "SparseReduce"]:        
        un, cts = np.unique(input, return_counts = True)
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        l_arr = np.array([i.size for i in un_idx])
        argsort = np.argsort(l_arr)
        un_sorted  = un[argsort]
        cts_sorted = cts[argsort]
        un_idx_sorted = [un_idx[i] for i in argsort]

        change = list(np.argwhere(l_arr[argsort][:-1] - l_arr[argsort][1:] != 0).flatten() + 1)
        change.insert(0, 0) 
        change.append(len(l_arr))
        change = np.array(change)

        el = []
        for i in range(len(change) - 1):
            el.append(np.array([un_idx_sorted[j] for j in range(change[i], change[i+1])]))

        #assert False
        return un_sorted, cts_sorted, SparseReduce(el, mean)

class BlockReduce(Reduce):
    """BlockReduce constructs a larger Gram matrix by copying indices of a smaller one

        Args:
            block_boundaries (np.array): Boundaries of blocks
            average (bool): wether or not to average
    """

    block_boundaries:Array
    average:bool = True
    def setup(self, ):
        if self.average:
            self._reduce = np.mean
        else:
            self._reduce = np.sum

    def reduce_first_ax(self, inp:np.array) -> np.array:
        assert (self.block_boundaries[-1]) <= len(inp), self.__class__.__name__ + " expects a longer gram to operate on"
        assert len(inp.shape) == 2
        rval = []

        for i in range(len(self.block_boundaries) - 1):
            start = self.block_boundaries[i]
            end = self.block_boundaries[i+1]
            reduced = self._reduce(inp[start:end,:], 0)
            rval.append(reduced)
        return np.concatenate(rval, 0)
    
    def new_len(self, original_len:int):
        return self.block_boundaries[-1]
    
    
    @classmethod
    def sum_from_block(cls, input:np.array, mean:bool = True) -> "BlockReduce":
        change = list(np.argwhere(input[:-1] - input[1:] != 0).flatten() + 1)
        change.insert(0, 0) 
        change.append(len(input))
        change = np.array(change)

        return BlockReduce(change, mean)

class LinearReduce(Reduce):
    linear_map:Array

    def reduce_first_ax(self, inp:np.array):
        assert len(inp.shape) == 2
        assert self.linear_map.shape[1] == inp.shape[0]
        return self.linear_map @ inp
    
    def new_len(self, original_len:int):
        assert (self.linear_map.shape[1]) == original_len, self.__class__.__name__ + " expects a gram with %d columns" % self.linear_map.shape[1]
        return self.linear_map.shape[0]

    @classmethod
    def sum_from_unique(cls, input:Array, mean:bool = True) -> Tuple[np.array, np.array, "LinearReduce"]:
        un, cts = np.unique(input, return_counts=True)
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        m = np.zeros((len(un_idx), input.shape[0]))
        for i, idx in enumerate(un_idx):
            b = np.ones(int(cts[i].squeeze())).squeeze()
            m = m.at[i, idx.squeeze()].set(b/cts[i].squeeze() if mean else b)
        return un, cts, LinearReduce(m)