import jax.numpy as np
from jaxrk.core.typing import Array, InitFn, Bijection
from dataclasses import dataclass

@dataclass
class ConstFn(InitFn):
    value:Array
    def __call__(self, shape=(1,), dtype = np.float32, bij:Bijection = None, ) -> Array:
        rval = np.ones(shape, dtype=dtype)
        if bij is not None:
            return rval * bij.inv(self.value)
        else:
            return rval * self.value

@dataclass
class ConstIsotropicFn(InitFn):
    value:Array
    def __call__(self, dim=1, dtype = np.float32, bij:Bijection = None, ) -> Array:
        rval = np.eye(dim, dtype=np.float32)
        if bij is not None:
            return rval * bij.inv(self.value)
        else:
            return rval * self.value