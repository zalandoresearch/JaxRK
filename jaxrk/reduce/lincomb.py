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


from .base import GramReduce

ListOfArray_or_Array_T = TypeVar("CombT", List[np.array], np.array)

class SparseReduce(GramReduce):
    def __init__(self, idcs:List[np.array], prefactors:ListOfArray_or_Array_T):
        """SparseReduce constructs a larger Gram matrix by copying indices of a smaller one

        Args:
            idcs (List[np.array]): The indices of the rows to copy in the desired order
            prefactors (ListOfArray_or_Array_T): Scalar prefactors for copied rows
        """
        super().__init__()
        self.idcs = idcs
        self.max_idx = np.max(np.array([np.max(i) for i in idcs]))
        self.prefactors = [(p if len(p.shape) == 2 else p[:, np.newaxis])
                                for p in prefactors]
    
    def reduce_first_ax(self, gram:np.array) -> np.array:
        assert (self.max_idx + 1) <= len(gram), self.__class__.__name__ + " expects a longer gram to operate on"
        rval = []
        for i, idx in enumerate(self.idcs):
            rval.append(np.sum(gram[idx] * self.prefactors[i], 0))
        return np.array(rval)
    
    def new_len(self, original_len:int):
        assert (self.max_idx + 1) <= original_len, self.__class__.__name__ + " expects a longer gram to operate on"
        return len(self.idcs)
    
    
    @classmethod
    def sum_from_unique(cls, input:np.array, mean:bool = True) -> (np.array, "SparseReduce"):        
        un, cts = np.unique(input, return_counts=True)
        if mean:
            pref = (1./cts.flatten())[:, np.newaxis, np.newaxis]
        else:
            pref = np.ones_like(cts.flatten())[:, np.newaxis, np.newaxis]
        un_idx = [np.argwhere(input == un[i]).flatten() for i in range(un.size)]
        return un, cts, SparseReduce(un_idx, pref)

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
    
   # @jit
    def reduce_first_ax(self, gram:np.array):
        assert self.linear_map.shape[1] == gram.shape[0]
        return self.linear_map @ gram
    
    def new_len(self, original_len:int):
        assert (self.linear_map.shape[1]) == original_len, self.__class__.__name__ + " expects a gram with %d columns" % self.linear_map.shape[1]
        return self.linear_map.shape[0]