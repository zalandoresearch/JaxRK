from abc import ABC, abstractmethod

from typing import NewType, TypeVar, Generic, Sized

class RkhsObject(object):
    pass

class Vec(RkhsObject, Sized):
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


class Map(RkhsObject, Generic[InpVecT, OutVecT]):
    """A vectorized map from InpVecT vectors to OutVecT vectors.
    """
    @abstractmethod
    def __len__(self):
        pass
    

