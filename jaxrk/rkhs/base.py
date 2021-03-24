from abc import ABC, abstractmethod
from jaxrk.core import Module
import flax.linen as ln
from typing import NewType, TypeVar, Generic, Sized, Union


class Vec(Sized, Module):
    @abstractmethod
    def reduce_gram(self, gram, axis = 0):
        pass
    
    @abstractmethod
    def inner(self, Y=None, full=True):
        pass

    @abstractmethod
    def __len__(self):
        pass

InpVecT = TypeVar("InpVecT", bound=Vec)
OutVecT = TypeVar("OutVecT", bound=Vec)

#The following is input to a map RhInpVectT -> InpVecT
RhInpVectT = TypeVar("RhInpVectT", bound=Vec) 

CombT = TypeVar("CombT", "LinOp[RhInpVectT, InpVecT]", InpVecT, np.array)


class LinOp(Vec, Generic[InpVecT, OutVecT]):
    @abstractmethod
    def __matmul__(self, right_inp:CombT) -> Union[OutVecT, "LinOp[RhInpVectT, OutVecT]"]:
        pass

