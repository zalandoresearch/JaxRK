from abc import abstractmethod, ABC
from ..core.typing import Array
import flax.linen as ln
from typing import NewType, TypeVar, Generic, Sized, Union


class Vec(Sized, ABC):
    @abstractmethod
    def reduce_gram(self, gram, axis = 0):
        pass
    
    @abstractmethod
    def inner(self, Y=None, full=True):
        pass

InpVecT = TypeVar("InpVecT", bound=Vec)
OutVecT = TypeVar("OutVecT", bound=Vec)

#The following is input to a map RhInpVectT -> InpVecT
RhInpVectT = TypeVar("RhInpVectT", bound=Vec) 

CombT = TypeVar("CombT", "LinOp[RhInpVectT, InpVecT]", InpVecT, Array)


class LinOp(Vec, Generic[InpVecT, OutVecT], ABC):
    @abstractmethod
    def __matmul__(self, right_inp:CombT) -> Union[OutVecT, "LinOp[RhInpVectT, OutVecT]"]:
        pass

RkhsObject = Union[Vec, LinOp]

