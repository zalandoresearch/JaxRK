from abc import ABC, abstractmethod


class RkhsObject(object):
    pass

class Vec(RkhsObject):
    @abstractmethod
    def reduce_gram(self, gram, axis = 0):
        pass
    
    @abstractmethod
    def inner(self, Y=None, full=True):
        pass

    @abstractmethod
    def __len__(self):
        pass

class Op(RkhsObject):    
    @abstractmethod
    def __len__(self):
        pass
    

