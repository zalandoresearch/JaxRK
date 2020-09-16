"""
Created on Thu Jan 10 10:01:56 2019

@author: Ingmar Schuster
"""


from typing import Callable, List, TypeVar

from jax import jit
import jax.numpy as np
import jax.scipy as sp
import jax.scipy.stats as stats
from jax.numpy import exp, log, sqrt
from jax.scipy.special import logsumexp
from jax.lax import scan, map
from jax import vmap


from .base import GramReduce

ListOfArray_or_Array_T = TypeVar("CombT", List[np.array], np.array)

class SparseReduce(GramReduce):
    def __init__(self, idcs:List[np.array], average:bool = True):
        """SparseReduce constructs a larger Gram matrix by copying indices of a smaller one

        Args:
            idcs (List[np.array]): The indices of the rows to copy in the desired order
            prefactors (ListOfArray_or_Array_T): Scalar prefactors for copied rows
        """
        super().__init__()
        self.idcs = idcs
        self.max_idx = np.max(np.array([np.max(i) for i in idcs]))
        if average:
            self._reduce = np.mean
        else:
            self._reduce = np.sum
    

    
    def reduce_first_ax(self, gram:np.array) -> np.array:
        assert (self.max_idx + 1) <= len(gram), self.__class__.__name__ + " expects a longer gram to operate on"
        #return map(lambda idx: self._reduce(gram[idx]), self.idcs)
        rval = np.zeros((len(self.idcs), gram.shape[1]))
        for i, idx in enumerate(self.idcs):
            rval = rval.at[i].set(self._reduce(gram[idx], 0))
        return rval
    
    def new_len(self, original_len:int):
        assert (self.max_idx + 1) <= original_len, self.__class__.__name__ + " expects a longer gram to operate on"
        return len(self.idcs)
    
    
    @classmethod
    def sum_from_unique(cls, input:np.array, mean:bool = True) -> (np.array, "SparseReduce"):        
        un, cts = np.unique(input, return_counts=True)
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        return un, cts, SparseReduce(un_idx, mean)

class LinearReduce(GramReduce):
    def __init__(self, linear_map:np.array):
        super().__init__()
        self.linear_map = linear_map
    
    @classmethod
    def sum_from_unique(cls, input:np.array, mean:bool = True) -> (np.array, "LinearReduce"):
        un, cts = np.unique(input, return_counts=True)
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        m = np.zeros((len(un_idx), input.shape[0]))
        for i, idx in enumerate(un_idx):
            b = np.ones(int(cts[i].squeeze())).squeeze()
            m = m.at[i, idx.squeeze()].set(b/cts[i].squeeze() if mean else b)
        return un, cts, LinearReduce(m)
    
   
    def reduce_first_ax(self, gram:np.array):
        assert self.linear_map.shape[1] == gram.shape[0]
        return self.linear_map @ gram
    
    def new_len(self, original_len:int):
        assert (self.linear_map.shape[1]) == original_len, self.__class__.__name__ + " expects a gram with %d columns" % self.linear_map.shape[1]
        return self.linear_map.shape[0]